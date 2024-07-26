use std::{cmp::Ordering, collections::BTreeSet, ops::{Deref, DerefMut}, sync::{Arc, Mutex}};

#[derive(Clone, Debug)]
pub struct ConnInner {
    in_node: NodeGene,
    out_node: NodeGene,
    weight: f32,
    enabled: bool,
    innov: u32,
}

#[derive(Clone, Debug)]
pub struct ConnGene(Arc<Mutex<ConnInner>>);

impl ConnGene {
    pub fn new(in_node: NodeGene, out_node: NodeGene, weight: f32, innov: u32) -> Self {
        Self(Arc::new(Mutex::new(ConnInner { in_node, out_node, weight, enabled: true, innov })))
    }

    pub fn in_node(&self) -> NodeGene {
        self.0.lock().unwrap().in_node.clone()
    }

    pub fn out_node(&self) -> NodeGene {
        self.0.lock().unwrap().out_node.clone()
    }

    pub fn weight(&self) -> f32 {
        self.0.lock().unwrap().weight.clone().into()
    }

    pub fn enabled(&self) -> bool {
        self.0.lock().unwrap().enabled
    }

    pub fn disable(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.enabled = false;
    }

    // these are for debug purposes
    pub fn innov(&self) -> u32 {
        self.0.lock().unwrap().innov
    }
}

impl Eq for ConnGene { }

impl Ord for ConnGene {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::ptr_eq(&self.0, &other.0)
            .then_some(Ordering::Equal)
            .unwrap_or(self.0.lock().unwrap().innov.cmp(&other.0.lock().unwrap().innov))
    }
}

impl PartialEq for ConnGene {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialOrd for ConnGene {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(&other).into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeKind {
    Input { forward: BTreeSet<ConnGene> },
    Hidden { forward: BTreeSet<ConnGene>, backward: BTreeSet<ConnGene> },
    Output { backward: BTreeSet<ConnGene> },
}

impl Ord for NodeKind {
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

impl PartialOrd for NodeKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(&other).into()
    }
}

#[derive(Clone, Debug)]
pub struct NodeGene(Arc<Mutex<NodeKind>>);

impl NodeGene {
    pub fn new_input() -> Self {
        Self(Arc::new(Mutex::new(NodeKind::Input { forward: Default::default() })))
    }
    
    pub fn new_hidden() -> Self {
        Self(Arc::new(Mutex::new(NodeKind::Hidden { forward: Default::default(), backward: Default::default() })))
    }

    pub fn new_output() -> Self {
        Self(Arc::new(Mutex::new(NodeKind::Output { backward: Default::default() })))
    }

    pub fn is_input_node(&self) -> bool {
        matches!(self.kind().lock().unwrap().deref(), NodeKind::Input { .. })
    }

    pub fn is_hidden_node(&self) -> bool {
        matches!(self.kind().lock().unwrap().deref(), NodeKind::Hidden { .. })
    }

    pub fn is_output_node(&self) -> bool {
        matches!(self.kind().lock().unwrap().deref(), NodeKind::Output { .. })
    }

    fn kind(&self) -> Arc<Mutex<NodeKind>> {
        self.0.clone()
    }

    pub fn forward<T>(&self, f: impl Fn(&BTreeSet<ConnGene>) -> T) -> Result<T, ()> {
        match self.0.lock().unwrap().deref() {
            NodeKind::Input { forward } | NodeKind::Hidden { forward, .. } => Ok(f(forward)),
            NodeKind::Output { .. } => Err(()),
        }
    }

    pub fn backward<T>(&self, f: impl Fn(&BTreeSet<ConnGene>) -> T) -> Result<T, ()> {
        match self.0.lock().unwrap().deref() {
            NodeKind::Input { .. } => Err(()),
            NodeKind::Hidden { backward, .. } | NodeKind::Output { backward } => Ok(f(backward)),
        }
    }

    pub fn forward_mut<T>(&self, mut f: impl FnMut(&mut BTreeSet<ConnGene>) -> T) -> Result<T, ()> {
        match self.0.lock().unwrap().deref_mut() {
            NodeKind::Input { forward } | NodeKind::Hidden { forward, .. } => Ok(f(forward)),
            NodeKind::Output { .. } => Err(()),
        }
    }

    pub fn backward_mut<T>(&self, mut f: impl FnMut(&mut BTreeSet<ConnGene>) -> T) -> Result<T, ()> {
        match self.0.lock().unwrap().deref_mut() {
            NodeKind::Input { .. } => Err(()),
            NodeKind::Hidden { backward, .. } | NodeKind::Output { backward } => Ok(f(backward)),
        }
    }
}

impl Eq for NodeGene { }

impl Ord for NodeGene {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.kind(), &other.kind()) {
            return Ordering::Equal;
        }

        let temp = self.backward(|backward| backward.len()).unwrap_or(0)
            .cmp(&other.backward(|backward| backward.len()).unwrap_or(0))
            .reverse();

        self.kind().lock().unwrap().cmp(&other.kind().lock().unwrap()).clone().then(temp)
    }
}

impl PartialEq for NodeGene {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.kind(), &other.kind())
    }
}

impl PartialOrd for NodeGene {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(&other).into()
    }
}