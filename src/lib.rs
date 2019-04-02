extern crate rand;
extern crate smallvec;

// Some copy and paste from https://github.com/zxqfl/mcts 

use std::collections::HashMap;
use rand::prelude::*;
use smallvec::SmallVec;

pub trait GameState where Self : Clone {
    type Move;
    type Player;
    type Eval;
    fn available_moves(&self) -> Vec<Self::Move>;
    fn rollout(&mut self, k: &mut rand::rngs::ThreadRng) { 
        loop {
            let m = self.available_moves();
            if m.is_empty() {
                break;
            }
            self.make_move(&m[k.gen_range(0, m.len())]);
        }
    }
    fn make_move(&mut self, mm:&Self::Move);
    fn current_player(&self) -> Self::Player;
    fn hash(&self) -> u64;
    fn evaluate(&self) -> Self::Eval;
    fn interpret_evaluation_for_player(&self, evaln: &Self::Eval, player: Self::Player) -> f64;
}

type ChildNodes = SmallVec::<[usize; 2]>;

struct MoveInfo<Move> {
    mv        : Move,
    children  : ChildNodes
}

#[derive(Default)]
struct Node<Move, Player> {
    hash: u64,
    moves: Option<Vec<MoveInfo<Move>>>,
    visits: u64,
    value: f64,
    player: Player,
}

pub struct MCTS<S:GameState> {
    root_state: S,
    nodes: Vec<Node<S::Move, S::Player>>,
    tracking: HashMap<u64, usize>,
    rng: rand::rngs::ThreadRng,
    exploration_constant: f64
}

impl<S:GameState> MCTS<S> where S::Move : Clone + std::fmt::Debug, S::Player : Clone {

    pub fn new(root_state:S, exploration_constant:f64) -> Self {
        MCTS {
            root_state,
            nodes: Vec::new(),
            tracking: HashMap::new(),
            rng: rand::thread_rng(),
            exploration_constant: exploration_constant
        }
    }

    pub fn playout_n(&mut self, count:usize) {
        self.tracking.clear();

        self.nodes.push(Node {
            hash: 0,
            moves: None,
            player: self.root_state.current_player(),
            visits: 0,
            value: 0.0
        });

        for _ in 0..count {
            self.playout();
        }

        println!(" ======== FINISHED {} playouts ======== ", count);
        let k = &self.nodes[0];
        println!("visits={}", k.visits);
        println!("value ={}", k.value);
        for m in k.moves.as_ref().unwrap() {            
            let mut vis = 0;
            let mut val = 0.0;
            for c in m.children.iter() {
                vis += self.nodes[*c].visits;
                val += self.nodes[*c].value;
            }
            //println!("   vis={}  val={}", vis, val);
            if vis > 0 {
                println!("{}. move {:?} has value {} ({} children)", vis, m.mv, val as f64 / vis as f64, m.children.len());
                let mut vk = Vec::new();
                self.add_best_moves(m.children[0], &mut vk);
                println!("       path={:?}", vk);
            } else {
                println!("{}. move {:?} was unexplored", vis, m.mv);
            }
        }
    }

    pub fn best_move(&self) -> Option<S::Move> {
        let k = &self.nodes[0];
        if let Some(ref moves) = k.moves {
            let mut rng = rand::thread_rng();
            select_by_key(&mut rng, moves.iter(), |m| {                
                m.children.iter().map(|c| self.nodes[*c].visits).sum::<u64>() as f64
            }).map(|best| best.mv.clone())
        } else {
            None
        }
    }

    pub fn add_best_moves(&self, src:usize, output:&mut Vec<S::Move>) {
        let k = &self.nodes[src];
        if let Some(ref moves) = k.moves {
            let mut rng = rand::thread_rng();
            if let Some(best) = select_by_key(&mut rng, moves.iter(), |m| {                
                m.children.iter().map(|c| self.nodes[*c].visits).sum::<u64>() as f64
            }) {
                output.push(best.mv.clone());
                self.add_best_moves(best.children[0], output);
            }
        }
    }   

    fn playout(&mut self) {        
        let mut state = self.root_state.clone();
        let mut pos = 0;
        let mut unexp = SmallVec::<[usize; 32]>::new();
        let mut steps_remaining = 1000;
        let mut path:Vec<usize> = Vec::new();

        while steps_remaining > 0 {
            path.push(pos);
            steps_remaining -= 1;
            let (next, hash, player) =
            {
                {   
                    // Expand moves and fill in unexp                
                    let node = &mut self.nodes[pos];
                    if node.moves.is_none() {
                        node.moves = Some(state.available_moves().into_iter().map(|x| MoveInfo { mv:x, children:SmallVec::new() }).collect());
                    }
                    
                    let moves = node.moves.as_ref().unwrap();
                    if moves.is_empty() {
                        break;
                    }

                    unexp.clear();
                    for p in moves.iter().enumerate() {
                        if p.1.children.is_empty() {
                            unexp.push(p.0);
                        }
                    }
                }

                let next = if unexp.len() > 0 {
                    // Randomly picked unexplored one.
                    unexp[self.rng.gen_range(0, unexp.len())]
                } else {
                    let mut total_visits = 0;
                    let mut pm = Vec::new();
                    for m in self.nodes[pos].moves.as_ref().unwrap().iter() {
                        let visits:u64 = m.children.iter().map(|x| self.nodes[*x].visits).sum();
                        let value:f64 = m.children.iter().map(|x| self.nodes[*x].value).sum();
                        let l = pm.len();
                        pm.push((visits, value, l));
                        total_visits += visits;
                    }    
                    let adjusted_total = (total_visits + 1) as f64;
                    let ln_adjusted_total = adjusted_total.ln();
                    let exp = self.exploration_constant;
                    let pick = select_by_key(&mut self.rng, pm.iter(), |mov| {
                        let sum_rewards = mov.1;
                        let child_visits = mov.0;
                        let explore_term = 2.0 * (ln_adjusted_total / child_visits as f64).sqrt();
                        let mean_action_value = sum_rewards as f64 / child_visits as f64;
                        exp * explore_term + mean_action_value
                    }).unwrap().2;
                    pick
                };

                let moves = self.nodes[pos].moves.as_ref().unwrap();
                let player = state.current_player();
                state.make_move(&moves[next].mv);
                let hash = state.hash();
                (next, hash, player)
            };

            let (new_pos, made_new) = if let Some(wh) = self.tracking.get(&hash).map(|x| *x) {
                assert_eq!(self.nodes[wh].hash, hash);
                (wh, false)
            } else {
                let new_pos = self.nodes.len();
                self.tracking.insert(hash, new_pos);
                self.nodes.push(Node {
                    hash,
                    moves: None,
                    player: player,
                    visits: 0,
                    value: 0.0
                });
                path.push(new_pos);
                (new_pos, true)
            };

            let mv = &mut self.nodes[pos].moves.as_mut().unwrap()[next];
            if !mv.children.contains(&new_pos) {
                mv.children.push(new_pos);
            }

            if made_new {
                break;
            }

            pos = new_pos;
        }
        state.rollout(&mut self.rng);
        let eval = state.evaluate();
        for p in path.iter().rev() {
            self.nodes[*p].visits += 1;
            self.nodes[*p].value += state.interpret_evaluation_for_player(&eval, self.nodes[*p].player.clone());
        }
    }
}

fn select_by_key<T, Iter, KeyFn>(rng: &mut rand::rngs::ThreadRng, elts: Iter, mut key_fn: KeyFn) -> Option<T> where Iter: Iterator<Item=T>, KeyFn: FnMut(&T) -> f64
{
    let mut choice = None;
    let mut num_optimal: u32 = 0;
    let mut best_so_far: f64 = std::f64::NEG_INFINITY;
    for elt in elts {
        let score = key_fn(&elt);
        if score > best_so_far {
            choice = Some(elt);
            num_optimal = 1;
            best_so_far = score;
        } else if score == best_so_far {
            num_optimal += 1;
            if rng.gen_range(0, num_optimal) == 0 {
                choice = Some(elt);
            }
        }
    }
    choice
}
