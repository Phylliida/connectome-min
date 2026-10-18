//! Autobiographical context compaction: a port of reimplclean2 (itself a clean reimplementation of
//! @animalabs/context-manager 0.10.1's AutobiographicalStrategy) over serde_json. A session lives
//! on disk as append-only JSONL logs (store.rs); a compile plans the context from prices and block
//! shapes only (emit::select) and an emit::Reader reads its content as the caller writes it out, so
//! neither the session nor a context is ever held in memory.

pub mod blocks;
pub mod chunking;
pub mod config;
pub mod drive;
pub mod emit;
pub mod estimate;
pub mod js;
pub mod json;
pub mod memory;
pub mod planner;
pub mod segment;
pub mod store;
pub mod turns;
