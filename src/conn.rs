use std::{cmp::Ordering, fmt::Debug, sync::{Arc, Mutex}};
use crate::traits;

#[derive(Clone, Debug)]
pub struct ConnInner<G: traits::Genome> {
    in_node: G::NodeGene,
    out_node: G::NodeGene,
    weight: f32,
    enabled: bool,
    innov: u32,
}

#[derive(Clone, Debug)]
pub struct ConnGene<G>(Arc<Mutex<ConnInner<G>>>)
where
    G: traits::Genome,
    G::NodeGene: Debug;

impl<G> ConnGene<G>
where
    G: traits::Genome,
    G::NodeGene: Debug
{
    pub fn new(in_node: G::NodeGene, out_node: G::NodeGene, weight: f32, innov: u32) -> Self {
        Self(Arc::new(Mutex::new(ConnInner { in_node, out_node, weight, enabled: true, innov })))
    }

    pub fn set_weight(&self, weight: f32) { self.0.lock().unwrap().weight = weight }

    pub fn set_enabled(&self, enabled: bool) {
        self.0.lock().unwrap().enabled = enabled;
    }
}

impl<G> traits::ConnGene<G> for ConnGene<G>
where
    G: traits::Genome,
    G::NodeGene: Clone + Debug
{
    fn in_node(&self) -> G::NodeGene { self.0.lock().unwrap().in_node.clone() }

    fn out_node(&self) -> G::NodeGene { self.0.lock().unwrap().out_node.clone() }

    fn weight(&self) -> f32 { self.0.lock().unwrap().weight.clone().into() }

    fn enabled(&self) -> bool { self.0.lock().unwrap().enabled }

    fn innov(&self) -> u32 { self.0.lock().unwrap().innov }
}

impl<G> Eq for ConnGene<G> where G: traits::Genome, G::NodeGene: Debug { }

impl<G> Ord for ConnGene<G> where G: traits::Genome, G::NodeGene: Debug {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::ptr_eq(&self.0, &other.0)
            .then_some(Ordering::Equal)
            .unwrap_or(self.0.lock().unwrap().innov.cmp(&other.0.lock().unwrap().innov))
    }
}

impl<G> PartialEq for ConnGene<G> where G: traits::Genome, G::NodeGene: Debug {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<G> PartialOrd for ConnGene<G> where G: traits::Genome, G::NodeGene: Debug {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(&other).into()
    }
}