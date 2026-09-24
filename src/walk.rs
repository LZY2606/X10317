//! Filesystem traversal built on the [`crate::glob_match`] engine.
//!
//! This module turns the pure string matcher into a real traversal chain:
//! patterns are compiled once into a [`GlobSet`], a walker descends an
//! abstract filesystem ([`FsView`]) and only descends a directory when a
//! structural prefix analysis proves a positive pattern can still match
//! inside it. Negated patterns never prune a subtree on their own —
//! re-inclusion is always possible.
//!
//! Two filesystems are provided:
//!
//! * [`StdFs`] is the production backend over the operating-system tree.
//! * [`VirtualTree`] is a small in-memory tree shared by the crate's own
//!   property suite. It deliberately supports symlink cycles, dangling
//!   links, unreadable nodes, case-colliding names, permuted read order and
//!   one-shot fault injection without depending on host permissions or the
//!   platform's enumeration order.
//!
//! All per-traversal state lives in the traversal session: compiling two
//! [`GlobSet`]s gives them distinct identities, and nothing from one
//! [`GlobSet::walk`] call leaks into the next.

use std::cell::Cell;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{validate, Error as PatternError, ErrorKind};

/// Metadata the walker needs for one node, independent of the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    File,
    Directory,
    Symlink,
}

/// One entry returned by [`FsView::read_dir`].
#[derive(Clone, Debug)]
pub struct DirEntry {
    /// The single component name, never containing a separator.
    pub name: String,
    pub kind: NodeKind,
}

/// What a symlink points at after backend-specific root translation.
#[derive(Clone, Debug)]
pub enum LinkTarget {
    /// A path the backend can continue traversing within the walk root.
    Path {
        /// Logical path in the walker's `/`-separated space.
        logical: String,
        /// Backend-specific handle used to reopen the target.
        handle: PathBuf,
        /// Whether the target escapes the walk root.
        escapes_root: bool,
    },
    /// The link exists but its target does not (a dangling link).
    Dangling,
}

/// Result of resolving a node that may itself be a symlink chain.
#[derive(Clone, Debug)]
pub struct ResolvedNode {
    pub kind: NodeKind,
    pub logical: String,
    pub handle: PathBuf,
}

/// Classification of a backend error at the walker's error boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A transient failure, such as an injected one-shot fault. Retried
    /// exactly once per operation by the production traversal logic.
    Transient,
    /// A node cannot be read (e.g. a permission denial). Recorded as a
    /// diagnostic; traversal continues with siblings.
    AccessDenied,
    /// A symlink chain reached [`WalkOptions::symlink_budget`].
    SymlinkBudget,
    /// A symlink chain revisited a node and was broken to avoid a loop.
    SymlinkLoop,
    /// A pattern rejected by [`validate`] or another invalid argument.
    InvalidPattern(PatternError),
    /// Any other backend failure that aborts traversal.
    Other,
}

/// A structured error produced by the traversal chain.
#[derive(Clone, Debug)]
pub struct WalkError {
    pub kind: ErrorKind,
    /// Logical path being operated on, if one was available.
    pub path: Option<String>,
    pub message: String,
}

impl WalkError {
    pub fn new(kind: ErrorKind, path: impl Into<Option<String>>, message: impl Into<String>) -> Self {
        Self { kind, path: path.into(), message: message.into() }
    }
}

impl std::fmt::Display for WalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.path {
            Some(path) => write!(f, "{} at {path}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for WalkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::InvalidPattern(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PatternError> for WalkError {
    fn from(error: PatternError) -> Self {
        Self::new(
            ErrorKind::InvalidPattern(error),
            None,
            format!("invalid glob pattern: {error}"),
        )
    }
}

/// Abstracts the filesystem boundary so the traversal logic is shared by the
/// production backend and the deterministic in-memory backend.
///
/// Paths arrive in the walker's logical `/`-separated space. `handle` carries
/// whatever the backend needs to reopen the node (an OS path for [`StdFs`],
/// the same logical path for [`VirtualTree`]).
pub trait FsView {
    /// Stat the node, following no symlinks.
    fn metadata(&self, logical: &str, handle: &Path) -> Result<ResolvedNode, WalkError>;

    /// List a directory's immediate entries in arbitrary order; the walker
    /// sorts the result and never relies on this order.
    fn read_dir(&self, logical: &str, handle: &Path) -> Result<Vec<DirEntry>, WalkError>;

    /// Read and translate a symlink target.
    fn open_link(&self, logical: &str, handle: &Path) -> Result<LinkTarget, WalkError>;


/// Production backend over the operating-system filesystem, rooted at `root`.
#[derive(Clone, Debug)]
pub struct StdFs {
    root: PathBuf,
}

impl StdFs {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: normalize_root(&root.into()) }
    }
}

fn io_kind(error: &std::io::Error) -> ErrorKind {
    match error.kind() {
        std::io::ErrorKind::PermissionDenied => ErrorKind::AccessDenied,
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted => ErrorKind::Transient,
        _ => ErrorKind::Other,
    }
}

/// Normalize a path, dropping `.` components and resolving `..` lexically
/// without touching the filesystem.
fn normalize_root(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

/// Translate an OS symlink target into a logical path relative to the walk
/// root, keeping the physical handle alongside it.
fn translate_target(
    target: &Path,
    parent_logical: &str,
    root: &Path,
) -> (String, PathBuf, bool) {
    let relative_parent = parent_logical.trim_start_matches('/');
    let physical = if target.is_absolute() {
        target.to_path_buf()
    } else {
        let parent = if relative_parent.is_empty() {
            root.to_path_buf()
        } else {
            root.join(relative_parent)
        };
        parent.join(target)
    };
    let normalized = normalize_root(&physical);

    match normalized.strip_prefix(root) {
        Ok(rel) => {
            let rendered = rel
                .components()
                .filter_map(|c| match c {
                    Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");
            let logical = if rendered.is_empty() { String::new() } else { format!("/{rendered}") };
            (logical, normalized, false)
        }
        Err(_) => (normalized.to_string_lossy().into_owned(), normalized, true),
    }
}

impl FsView for StdFs {
    fn metadata(&self, _logical: &str, handle: &Path) -> Result<ResolvedNode, WalkError> {
        let metadata = std::fs::symlink_metadata(handle).map_err(|error| WalkError {
            kind: io_kind(&error),
            path: Some(handle.to_string_lossy().into_owned()),
            message: error.to_string(),
        })?;

        let kind = if metadata.file_type().is_symlink() {
            NodeKind::Symlink
        } else if metadata.is_dir() {
            NodeKind::Directory
        } else {
            NodeKind::File
        };

        Ok(ResolvedNode { kind, logical: String::new(), handle: handle.to_path_buf() })
    }

    fn read_dir(&self, _logical: &str, handle: &Path) -> Result<Vec<DirEntry>, WalkError> {
        let entries = std::fs::read_dir(handle).map_err(|error| WalkError {
            kind: io_kind(&error),
            path: Some(handle.to_string_lossy().into_owned()),
            message: error.to_string(),
        })?;

        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| WalkError {
                kind: io_kind(&error),
                path: Some(handle.to_string_lossy().into_owned()),
                message: error.to_string(),
            })?;

            let file_type = entry.file_type().map_err(|error| WalkError {
                kind: io_kind(&error),
                path: Some(handle.to_string_lossy().into_owned()),
                message: error.to_string(),
            })?;

            let kind = if file_type.is_symlink() {
                NodeKind::Symlink
            } else if file_type.is_dir() {
                NodeKind::Directory
            } else {
                NodeKind::File
            };

            out.push(DirEntry { name: entry.file_name().to_string_lossy().into_owned(), kind });
        }
        Ok(out)
    }

    fn open_link(&self, logical: &str, handle: &Path) -> Result<LinkTarget, WalkError> {
        let target = std::fs::read_link(handle).map_err(|error| WalkError {
            kind: io_kind(&error),
            path: Some(logical.to_owned()),
            message: error.to_string(),
        })?;

        let (translated_logical, physical, escapes_root) =
            translate_target(&target, logical, &self.root);

        match std::fs::symlink_metadata(&physical) {
            Ok(_) => Ok(LinkTarget::Path {
                logical: translated_logical,
                handle: physical,
                escapes_root,
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(LinkTarget::Dangling)
            }
            Err(error) => Err(WalkError {
                kind: io_kind(&error),
                path: Some(translated_logical),
                message: error.to_string(),
            }),
        }
    }
}

/// Traversal configuration. Construct with [`WalkOptions::default`].
#[derive(Clone, Debug)]
pub struct WalkOptions {
    /// ASCII case-insensitive matching when `false`. Defaults to `true`.
    pub case_sensitive: bool,
    /// Separator understood by patterns and logical paths. Normalized to
    /// `/` internally; defaults to `/`.
    pub separator: u8,
    /// Maximum number of symlink hops in one chain before the chain is
    /// dropped with [`ErrorKind::SymlinkBudget`]. Defaults to 16.
    pub symlink_budget: usize,
    /// Include matching directories in the result, not only files.
    pub include_directories: bool,
    /// Whether dangling symlinks are eligible matches (a dangling link has
    /// no node kind beyond the link itself). Defaults to `true`.
    pub include_dangling_links: bool,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            separator: b'/',
            symlink_budget: 16,
            include_directories: false,
            include_dangling_links: true,
        }
    }
}

/// A single pattern after negation stripping and separator normalization.
struct CompiledPattern {
    /// Pattern text with leading `!`s stripped, separator-normalized.
    text: Vec<u8>,
    negated: bool,
    /// Structural segments used for the over-approximating prefix test.
    segments: Vec<PatternSegment>,
    /// True when segments could not be derived; the pattern is then treated
    /// as potentially matching everywhere (never pruned).
    universal: bool,
}

#[derive(Clone)]
enum PatternSegment {
    /// Literal characters, case-folded when matching is case-insensitive.
    Literal(Vec<u8>),
    /// `*` or `?` or a character class within one segment.
    SingleSegment,
    /// `**` — spans any number of segments.
    Globstar,
}

/// Split a pattern into structural segments. Returns `None` when a construct
/// (a brace expansion or unclosed class) makes static analysis unsafe; the
/// caller then treats the pattern as universal so it can never be pruned.
fn split_segments(pattern: &[u8]) -> Option<Vec<PatternSegment>> {
    let mut segments: Vec<PatternSegment> = vec![PatternSegment::Literal(Vec::new())];
    let mut brace_depth: usize = 0;
    let mut index = 0;

    while index < pattern.len() {
        match pattern[index] {
            b'\\' => {
                if index + 1 >= pattern.len() {
                    return None;
                }
                if let PatternSegment::Literal(bytes) = segments.last_mut().unwrap() {
                    bytes.push(pattern[index + 1]);
                }
                index += 2;
            }
            b'[' => {
                // Skip the class; treat the segment as a one-segment wildcard.
                let mut class = index + 1;
                if matches!(pattern.get(class), Some(b'^' | b'!')) {
                    class += 1;
                }
                let mut first = true;
                loop {
                    match pattern.get(class)? {
                        b']' if !first => break,
                        b'\\' => class += 1,
                        _ => {}
                    }
                    first = false;
                    class += 1;
                }
                *segments.last_mut().unwrap() = PatternSegment::SingleSegment;
                index = class + 1;
            }
            b'{' => return None,
            b'}' | b',' if brace_depth > 0 => return None,
            b'/' => {
                segments.push(PatternSegment::Literal(Vec::new()));
                index += 1;
            }
            b'*' if pattern.get(index + 1) == Some(&b'*') => {
                // `**` only spans segments when bounded by slashes/end.
                let bounded_left = index == 0 || pattern[index - 1] == b'/';
                let after = index + 2;
                let bounded_right = after == pattern.len() || pattern[after] == b'/';
                if bounded_left && bounded_right {
                    *segments.last_mut().unwrap() = PatternSegment::Globstar;
                    if after < pattern.len() {
                        segments.push(PatternSegment::Literal(Vec::new()));
                        index = after + 1;
                        continue;
                    }
                    index = after;
                } else {
                    *segments.last_mut().unwrap() = PatternSegment::SingleSegment;
                    index += 2;
                }
            }
            b'*' | b'?' => {
                *segments.last_mut().unwrap() = PatternSegment::SingleSegment;
                index += 1;
            }
            byte => {
                if let PatternSegment::Literal(bytes) = segments.last_mut().unwrap() {
                    bytes.push(byte);
                }
                index += 1;
            }
        }
    }

    Some(segments)
}

fn fold_case(byte: u8, case_sensitive: bool) -> u8 {
    if !case_sensitive && byte.is_ascii_uppercase() {
        byte.to_ascii_lowercase()
    } else {
        byte
    }
}

impl CompiledPattern {
    fn compile(raw: &str, options: &WalkOptions) -> Result<Self, WalkError> {
        let bytes = raw.as_bytes();
        let mut negated = false;
        let mut offset = 0;
        while offset < bytes.len() && bytes[offset] == b'!' {
            negated = !negated;
            offset += 1;
        }

        let mut text: Vec<u8> = bytes[offset..].to_vec();
        validate(&text).map_err(WalkError::from)?;

        if options.separator != b'/' {
            for byte in &mut text {
                if *byte == options.separator {
                    *byte = b'/';
                }
            }
        }

        let segments = split_segments(&text).unwrap_or_else(|| vec![PatternSegment::Globstar]);
        let universal = split_segments(&text).is_none();
        let segments = segments
            .into_iter()
            .map(|segment| match segment {
                PatternSegment::Literal(mut bytes) => {
                    for byte in &mut bytes {
                        *byte = fold_case(*byte, options.case_sensitive);
                    }
                    PatternSegment::Literal(bytes)
                }
                other => other,
            })
            .collect();

        Ok(Self { text, negated, segments, universal })
    }
}

static GLOB_SET_SEQ: AtomicU64 = AtomicU64::new(1);

/// Compiled positive and negative patterns with shared traversal settings.
#[derive(Debug)]
pub struct GlobSet {
    patterns: Vec<CompiledPattern>,
    options: WalkOptions,
    identity: u64,
}

impl GlobSet {
    /// Compile patterns in order; the last matching pattern wins, so a
    /// negative pattern after a positive one excludes and vice versa.
    pub fn new<'a, I>(patterns: I, options: WalkOptions) -> Result<Self, WalkError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut compiled = Vec::new();
        for pattern in patterns {
            compiled.push(CompiledPattern::compile(pattern, &options)?);
        }
        Ok(Self {
            patterns: compiled,
            options,
            identity: GLOB_SET_SEQ.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Distinct per-compiled-set identity; a new compile is a new version,
    /// so cached traversal decisions can never be shared across versions.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Traverse `fs` rooted at logical `""` with `root_handle` as the backend
    /// handle of the root directory.
    pub fn walk<F: FsView>(
        &self,
        fs: &F,
        root_handle: &Path,
    ) -> Result<WalkOutcome, WalkError> {
        let mut session = Session::new(self, fs, root_handle);
        session.run()
    }
}

/// One node on the DFS stack.
struct Frame {
    logical: String,
    handle: PathBuf,
    /// Logical targets seen on the symlink chain that opened this frame.
    chain_seen: Vec<String>,
    /// Symlink hops already taken on this frame's chain.
    chain_hops: usize,
}

/// Counters describing one traversal; every traversal starts at zero.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WalkStats {
    pub set_identity: u64,
    pub dirs_opened: usize,
    pub dirs_pruned: usize,
    pub entries_seen: usize,
    pub symlinks_followed: usize,
    pub symlink_loops: usize,
    pub symlink_budget_exceeded: usize,
    pub dangling_links: usize,
    pub access_denied: usize,
    pub transient_retried: usize,
    pub transient_failed: usize,
    pub prune_cache_hits: usize,
}

/// A path emitted by a traversal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchedPath {
    /// `/`-separated logical path relative to the walk root (no leading `/`).
    pub path: String,
    pub kind: NodeKind,
}

/// The result of one traversal: sorted matches and sorted diagnostics.
#[derive(Clone, Debug, Default)]
pub struct WalkOutcome {
    pub matches: Vec<MatchedPath>,
    pub diagnostics: Vec<WalkError>,
    pub stats: WalkStats,
}

struct Session<'a, F: FsView> {
    set: &'a GlobSet,
    fs: &'a F,
    root_handle: PathBuf,
    stack: Vec<Frame>,
    outcome: WalkOutcome,
    /// Per-session prune cache, keyed by logical path. Version stamped with
    /// the compiled-set identity; the cache dies with the session.
    prune_cache: Vec<(String, u64, bool)>,
    remaining_faults: usize,
}

fn join_logical(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}/{name}")
    }
}

/// Case-fold a logical path the same way compiled patterns were folded.
fn fold_path(path: &[u8], case_sensitive: bool) -> Vec<u8> {
    if case_sensitive {
        path.to_vec()
    } else {
        path.iter().map(|byte| fold_case(*byte, false)).collect()
    }
}

/// Case-fold an entire pattern: literals, escaped characters and the
/// members/ranges of character classes. Combined with folding the path at
/// match time this keeps the production `glob_match` engine unchanged while
/// making matching case-insensitive. Brace syntax is unaffected (folding
/// never creates or removes braces).
fn fold_pattern(pattern: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pattern.len());
    let mut index = 0;
    while index < pattern.len() {
        let byte = pattern[index];
        match byte {
            b'\\' => {
                out.push(b'\\');
                if let Some(&next) = pattern.get(index + 1) {
                    out.push(fold_case(next, false));
                    index += 2;
                } else {
                    index += 1;
                }
            }
            b'[' => {
                out.push(b'[');
                let mut scan = index + 1;
                if matches!(pattern.get(scan), Some(b'^' | b'!')) {
                    out.push(pattern[scan]);
                    scan += 1;
                }
                let mut first = true;
                while scan < pattern.len() {
                    match pattern[scan] {
                        b']' if !first => break,
                        b'\\' => {
                            out.push(b'\\');
                            if let Some(&next) = pattern.get(scan + 1) {
                                out.push(fold_case(next, false));
                                scan += 2;
                            } else {
                                scan += 1;
                            }
                        }
                        other => {
                            out.push(fold_case(other, false));
                            scan += 1;
                        }
                    }
                    first = false;
                }
                if scan < pattern.len() {
                    out.push(b']');
                    index = scan + 1;
                } else {
                    index = scan;
                }
            }
            _ => {
                out.push(fold_case(byte, false));
                index += 1;
            }
        }
    }
    out
}
