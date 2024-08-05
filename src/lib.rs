mod config;
mod genes;
mod genome;

pub use config::Config;
pub use genes::{ConnGene, NodeGene};
pub use genome::FeedForwardGenome;

#[cfg(test)]
mod tests {
    use super::*;

    const GENOME: FeedForwardGenome = FeedForwardGenome::minimal::<3, 1>();

    #[test]
    fn mutate_add_conn() {
        let mut genome = GENOME;
        let in_node = genome.input.first().cloned().unwrap();
        let out_node = genome.output.first().cloned().unwrap();
        let new_conn = genome.add_conn(in_node, out_node, 0.5);
        assert_eq!(new_conn, genome.conns.first().cloned().unwrap());
    }
}
