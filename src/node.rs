use std::{cmp::Ordering, collections::BTreeSet, ops::{Deref, DerefMut}, sync::{Arc, Mutex}};
use crate::traits;

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind<G: traits::Genome> {
    Input { forward: BTreeSet<G::ConnGene> },
    Hidden { forward: BTreeSet<G::ConnGene>, backward: BTreeSet<G::ConnGene> },
    Output { backward: BTreeSet<G::ConnGene> },
}

impl<G: traits::Genome> Eq for NodeKind<G> { }

impl<G: traits::Genome> Ord for NodeKind<G> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Input { .. }, Self::Hidden { .. }) |
                (Self::Input { .. }, Self::Output { .. }) |
                (Self::Hidden { .. }, Self::Output { .. }) => Ordering::Greater,
            (Self::Hidden { .. }, Self::Input { .. }) |
                (Self::Output { .. }, Self::Input { .. }) |
                (Self::Output { .. }, Self::Hidden { .. }) => Ordering::Less,
            _ => Ordering::Equal
        }
    }
}

impl<G: traits::Genome> PartialOrd for NodeKind<G> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(&other).into()
    }
}

#[derive(Clone, Debug)]
pub struct NodeGene<G: traits::Genome>(Arc<Mutex<NodeKind<G>>>);

impl<G: traits::Genome> NodeGene<G> {
    pub fn new_input() -> Self {
        Self(Arc::new(Mutex::new(NodeKind::Input { forward: Default::default() })))
    }
    
    pub fn new_hidden() -> Self {
        Self(Arc::new(Mutex::new(NodeKind::Hidden { forward: Default::default(), backward: Default::default() })))
    }

    pub fn new_output() -> Self {
        Self(Arc::new(Mutex::new(NodeKind::Output { backward: Default::default() })))
    }

    fn kind(&self) -> Arc<Mutex<NodeKind<G>>> {
        self.0.clone()
    }

    pub fn forward<T>(&self, f: impl Fn(&BTreeSet<G::ConnGene>) -> T) -> T {
        match self.0.lock().unwrap().deref() {
            NodeKind::Input { forward } | NodeKind::Hidden { forward, .. } => f(forward),
            NodeKind::Output { .. } => f(&Default::default()),
        }
    }

    pub fn backward<T>(&self, f: impl Fn(&BTreeSet<G::ConnGene>) -> T) -> T {
        match self.0.lock().unwrap().deref() {
            NodeKind::Input { .. } => f(&mut Default::default()),
            NodeKind::Hidden { backward, .. } | NodeKind::Output { backward } => f(backward),
        }
    }

    pub fn forward_mut<T>(&self, mut f: impl FnMut(&mut BTreeSet<G::ConnGene>) -> T) -> T {
        match self.0.lock().unwrap().deref_mut() {
            NodeKind::Input { forward } | NodeKind::Hidden { forward, .. } => f(forward),
            NodeKind::Output { .. } => f(&mut Default::default()),
        }
    }

    pub fn backward_mut<T>(&self, mut f: impl FnMut(&mut BTreeSet<G::ConnGene>) -> T) -> T {
        match self.0.lock().unwrap().deref_mut() {
            NodeKind::Input { .. } => f(&mut Default::default()),
            NodeKind::Hidden { backward, .. } | NodeKind::Output { backward } => f(backward),
        }
    }
}

impl<G: traits::Genome> Eq for NodeGene<G> { }

impl<G: traits::Genome> Ord for NodeGene<G> {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.kind(), &other.kind()) {
            return Ordering::Equal;
        }

        let temp = self.backward(|backward| backward.len())
            .cmp(&other.backward(|backward| backward.len()))
            .reverse();

        self.kind().lock().unwrap().cmp(&other.kind().lock().unwrap()).clone().then(temp)
    }
}

impl<G: traits::Genome> PartialEq for NodeGene<G> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.kind(), &other.kind())
    }
}

impl<G: traits::Genome> PartialOrd for NodeGene<G> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(&other).into()
    }
}

impl<G: traits::Genome> traits::NodeGene for NodeGene<G> { }