use std::{cell::RefCell, cmp::Ordering, collections::BTreeSet, iter, rc::Rc};

#[derive(Debug)]
pub struct ConnGene {
    in_node: Rc<RefCell<NodeGene>>,
    out_node: Rc<RefCell<NodeGene>>,
    weight: f32,
    enabled: bool,
    innov: u32,
}

impl ConnGene {
    pub const fn new(
        in_node: Rc<RefCell<NodeGene>>,
        out_node: Rc<RefCell<NodeGene>>,
        weight: f32,
        innov: u32,
    ) -> Self {
        Self {
            in_node,
            out_node,
            weight,
            enabled: true,
            innov,
        }
    }

    pub fn in_node(&self) -> Rc<RefCell<NodeGene>> {
        Rc::clone(&self.in_node)
    }

    pub fn out_node(&self) -> Rc<RefCell<NodeGene>> {
        Rc::clone(&self.out_node)
    }

    pub const fn weight(&self) -> f32 {
        self.weight
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

impl Eq for ConnGene {}

impl Ord for ConnGene {
    fn cmp(&self, other: &Self) -> Ordering {
        self.innov.cmp(&other.innov)
    }
}

impl PartialEq for ConnGene {
    fn eq(&self, other: &Self) -> bool {
        self.innov.eq(&other.innov)
    }
}

impl PartialOrd for ConnGene {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.innov.partial_cmp(&other.innov)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeGene {
    Input {
        forward: BTreeSet<Rc<RefCell<ConnGene>>>,
    },
    Hidden {
        forward: BTreeSet<Rc<RefCell<ConnGene>>>,
        backward: BTreeSet<Rc<RefCell<ConnGene>>>,
    },
    Output {
        backward: BTreeSet<Rc<RefCell<ConnGene>>>,
    },
}

impl NodeGene {
    pub const fn new_input() -> Self {
        Self::Input {
            forward: BTreeSet::new(),
        }
    }

    pub const fn new_hidden() -> Self {
        Self::Hidden {
            forward: BTreeSet::new(),
            backward: BTreeSet::new(),
        }
    }

    pub const fn new_output() -> Self {
        Self::Output {
            backward: BTreeSet::new(),
        }
    }

    pub fn forward_conns(&self) -> usize {
        match self {
            Self::Input { forward } | Self::Hidden { forward, .. } => forward.len(),
            Self::Output { .. } => 0,
        }
    }

    pub fn backward_conns(&self) -> usize {
        match self {
            Self::Input { .. } => 0,
            Self::Hidden { backward, .. } | Self::Output { backward } => backward.len(),
        }
    }

    pub fn insert_forward_conn(&mut self, conn: Rc<RefCell<ConnGene>>) {
        match self {
            Self::Input { ref mut forward }
            | Self::Hidden {
                ref mut forward, ..
            } => {
                forward.insert(conn);
            }
            Self::Output { .. } => (),
        }
    }

    pub fn insert_backward_conn(&mut self, conn: Rc<RefCell<ConnGene>>) {
        match self {
            Self::Input { .. } => (),
            Self::Hidden {
                ref mut backward, ..
            }
            | Self::Output { ref mut backward } => {
                backward.insert(conn);
            }
        }
    }

    pub fn iter_forward_conns(&self) -> Box<dyn Iterator<Item = Rc<RefCell<ConnGene>>> + '_> {
        match self {
            Self::Input { forward } | Self::Hidden { forward, .. } => {
                Box::new(forward.iter().cloned())
            }
            Self::Output { .. } => Box::new(iter::empty()),
        }
    }

    pub fn iter_backward_conns(&self) -> Box<dyn Iterator<Item = Rc<RefCell<ConnGene>>> + '_> {
        match self {
            Self::Input { .. } => Box::new(iter::empty()),
            Self::Hidden { backward, .. } | Self::Output { backward } => {
                Box::new(backward.iter().cloned())
            }
        }
    }
}

impl Eq for NodeGene {}

impl Ord for NodeGene {
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self, &other) {
            (Self::Input { .. }, _) => Ordering::Greater,
            (Self::Output { .. }, _) => Ordering::Less,
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for NodeGene {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
