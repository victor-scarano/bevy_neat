use crate::{ConnGene, NodeGene};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use std::{
    cell::{LazyCell, RefCell},
    collections::{BTreeMap, BTreeSet},
    iter,
    num::Saturating,
    rc::Rc,
};

#[derive(Debug)]
pub struct FeedForwardGenome {
    pub(super) conns: BTreeSet<Rc<RefCell<ConnGene>>>,
    pub(super) input: LazyCell<Box<[Rc<RefCell<NodeGene>>]>>,
    pub(super) hidden: BTreeSet<Rc<RefCell<NodeGene>>>,
    pub(super) output: LazyCell<Box<[Rc<RefCell<NodeGene>>]>>,
}

impl FeedForwardGenome {
    pub const fn minimal<const INPUT_LEN: usize, const OUTPUT_LEN: usize>() -> Self {
        // TODO: Assert that input len and output len are non-zero.

        Self {
            conns: BTreeSet::new(),
            input: LazyCell::new(|| {
                iter::repeat_with(|| Rc::new(RefCell::new(NodeGene::new_input())))
                    .take(INPUT_LEN)
                    .collect()
            }),
            hidden: BTreeSet::new(),
            output: LazyCell::new(|| {
                iter::repeat(Rc::new(RefCell::new(NodeGene::new_output())))
                    .take(OUTPUT_LEN)
                    .collect()
            }),
        }
    }

    pub(super) fn add_conn(
        &mut self,
        in_node: Rc<RefCell<NodeGene>>,
        out_node: Rc<RefCell<NodeGene>>,
        weight: f32,
    ) -> Rc<RefCell<ConnGene>> {
        // TODO: Assert that the in node and out node passed to this function are part of this genome.

        let new_conn = Rc::new(RefCell::new(ConnGene::new(
            Rc::clone(&in_node),
            Rc::clone(&out_node),
            weight,
            u32::MAX,
        )));

        RefCell::borrow_mut(&Rc::clone(&in_node)).insert_forward_conn(Rc::clone(&new_conn));
        RefCell::borrow_mut(&Rc::clone(&in_node)).insert_backward_conn(Rc::clone(&new_conn));

        self.conns.insert(Rc::clone(&new_conn));

        Rc::clone(&new_conn)
    }

    pub(crate) fn add_node(&mut self, old_conn: Rc<RefCell<ConnGene>>) -> Rc<RefCell<NodeGene>> {
        // TODO: Assert that the old conn passed to this function are part of this genome.

        RefCell::borrow_mut(&old_conn).disable();

        let new_node = Rc::new(RefCell::new(NodeGene::new_hidden()));

        let conn_a = Rc::new(RefCell::new(ConnGene::new(
            RefCell::borrow(&old_conn).in_node(),
            Rc::clone(&new_node),
            1.0,
            u32::MAX,
        )));

        let conn_b = Rc::new(RefCell::new(ConnGene::new(
            Rc::clone(&new_node),
            RefCell::borrow(&old_conn).out_node(),
            RefCell::borrow(&old_conn).weight(),
            u32::MAX,
        )));

        RefCell::borrow_mut(&RefCell::borrow(&old_conn).in_node())
            .insert_forward_conn(Rc::clone(&conn_a));
        RefCell::borrow_mut(&RefCell::borrow(&old_conn).out_node())
            .insert_backward_conn(Rc::clone(&conn_b));

        RefCell::borrow_mut(&new_node).insert_forward_conn(Rc::clone(&conn_b));
        RefCell::borrow_mut(&new_node).insert_backward_conn(Rc::clone(&conn_a));

        self.conns.insert(Rc::clone(&conn_a));
        self.conns.insert(Rc::clone(&conn_b));

        self.hidden.insert(Rc::clone(&new_node));

        Rc::clone(&new_node)
    }

    pub fn mutate_add_conn(&mut self, mut rng: impl Rng) -> Rc<RefCell<ConnGene>> {
        let in_nodes = {
            let input_in_nodes = self.input.iter();
            let hidden_in_nodes = self.hidden.iter();
            let mut in_nodes: Vec<_> = input_in_nodes.chain(hidden_in_nodes).cloned().collect();
            in_nodes.shuffle(&mut rng);
            assert_ne!(in_nodes.len(), 0);
            in_nodes
        };

        let out_nodes = {
            let hidden_out_nodes = self.hidden.iter();
            let output_out_nodes = self.output.iter();
            let mut out_nodes: Vec<_> = hidden_out_nodes.chain(output_out_nodes).cloned().collect();
            out_nodes.shuffle(&mut rng);
            assert_ne!(out_nodes.len(), 0);
            out_nodes
        };

        let in_node = in_nodes
            .into_iter()
            .find(|node| {
                let mut count = Saturating(out_nodes.iter().count());
                count -= self.hidden.contains(&Rc::clone(node)) as usize;
                count -= RefCell::borrow(node).forward_conns();
                count.0 > 0
            })
            .expect("it has already been asserted that a valid in node exists");

        let out_node = out_nodes
            .into_iter()
            .find(|node| {
                !RefCell::borrow(node)
                    .iter_backward_conns()
                    .any(|conn| RefCell::borrow(&conn).in_node() == in_node)
            })
            .expect("it has already been asserted that a valid out node exists");

        self.add_conn(Rc::clone(&in_node), Rc::clone(&out_node), rng.gen())
    }

    pub fn mutate_add_node(&mut self, mut rng: impl Rng) -> Rc<RefCell<NodeGene>> {
        assert_ne!(self.conns.len(), 0);

        let old_conn = self
            .conns
            .iter()
            .filter(|conn| RefCell::borrow(&conn).enabled())
            .cloned()
            .choose(&mut rng)
            .expect("there should be at least one enabled connection gene present");

        self.add_node(old_conn)
    }

    pub fn mutate_conn_weight(&mut self) {
        todo!();
    }

    pub fn activate(&self, input: Vec<f32>) -> Vec<f32> {
        assert_eq!(input.len(), self.input.len());

        let mut nodes = BTreeMap::from_iter(
            self.input
                .iter()
                .enumerate()
                .map(|(idx, node)| (Rc::clone(&node), input.get(idx).cloned().unwrap())),
        );

        // TODO: Check if `BTreeSet::pop_first` or `BTreeSet::pop_last` is the right method for this use case.
        while let Some((node, val)) = nodes.pop_first() {
            // TODO: Apply activation function on value.

            for conn in RefCell::borrow(&node)
                .iter_forward_conns()
                .filter(|conn| RefCell::borrow(&conn).enabled())
            {
                let new_value = nodes.entry(RefCell::borrow(&conn).out_node()).or_default();
                *new_value += val * RefCell::borrow(&conn).weight()
            }
        }

        (0..self.output.len())
            .map(|idx| {
                let node = self.output.get(idx).unwrap();
                let val = nodes.get(node).cloned().unwrap();
                // TODO: Apply activation function on value.
                val
            })
            .collect()
    }

    pub fn comp_dist(_lhs: &Self, _rhs: &Self) -> f32 {
        todo!();
    }

    pub fn crossover(_lhs: Self, _rhs: Self) -> Self {
        todo!();
    }
}
