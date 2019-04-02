extern crate stmcts;
extern crate rand;

use stmcts::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hasher, Hash};

#[derive(Clone)]
struct CountingGame {
    turn: i64,
    value: i64,
    random: u64,
    used_mega: bool,
}

#[derive(Clone, Debug)]
enum Move {
    Add, 
    Sub,
    BigAdd,
    BigSub,
    MegaMove, // a trick move.
}

impl GameState for CountingGame {    
    type Move = (u64, Move);
    type Eval = i64;    
    type Player = ();

    fn current_player(&self) -> Self::Player {
        ()
    }

    fn available_moves(&self) -> Vec<Self::Move> {
        if self.turn == 50 {
            vec![]
        } else {
            let mut moves = Vec::new();
            let hash = self.hash();
            if self.random & 1 == 1 && !self.used_mega {
                moves.push((hash, Move::Add));
            }
            if self.random & 2 == 2 {
                moves.push((hash, Move::Sub));
            }
            if self.random & 4 == 4 && !self.used_mega {
                moves.push((hash, Move::BigAdd));
            }
            if self.random & 8 == 8 {
                moves.push((hash, Move::BigSub));
            }
            if !self.used_mega {
                moves.push((hash, Move::MegaMove));
            }
            moves
        }
    }

    fn evaluate(&self) -> Self::Eval {
        self.value
    }

    fn interpret_evaluation_for_player(&self, _evaln: &Self::Eval, _player: Self::Player) -> i64 {
        self.value
    }

    fn make_move(&mut self, mov: &Self::Move) {
        let hash = self.hash();
        if hash != mov.0 {
            println!("trying move {:?} on state {}", mov, hash);
        }
        assert_eq!(hash, mov.0);
        match mov.1 {
            Move::Add => self.value += 1,
            Move::Sub => self.value -= 1,
            Move::BigAdd => self.value += 5,
            Move::BigSub => self.value -= 5,
            Move::MegaMove => { self.value += 20; self.used_mega = true; }
        }
        self.turn += 1;
        self.random = rand::random::<u64>() & 0xf;        
    }

    fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.turn.hash(&mut hasher);
        self.value.hash(&mut hasher);
        self.random.hash(&mut hasher);
        self.used_mega.hash(&mut hasher);
        hasher.finish()
    }
}

fn main() {    
    let mut mcts = MCTS::new(CountingGame {
        value: 0,
        turn: 0,
        random: 0xf,
        used_mega: false
    });
    mcts.playout_n(10000);
}
