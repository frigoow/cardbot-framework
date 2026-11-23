# cardbot-framework: design

A header-only C++ library and a Python package, in the spirit of
[ismcsolver](https://github.com/sfranzen/ismcsolver): a game implements one
interface, and gets the pipeline that made the Manille and Belote bots
strong. A new game is still real work (its rules, its features), but none of
the search, sampling, training or evaluation has to be written again.

## The pipeline for a new game

| Step | The game writes | The framework provides | Out |
|---|---|---|---|
| 1. Rules, legal moves, states | `Game`: deal, the seat to move, legal actions, apply, the end and the score; `observe(seat)`: what a seat knows | the interface, card sets and decks, trick resolution helpers, a rules test harness (random games, replay of logs from another engine) | a playable game |
| 2. Baselines | optionally a hand-written bidder (otherwise random) | `RandomAgent`, `IsmctsAgent` (ismcsolver over the `Game`), a baseline bidder hook | baseline players |
| 3. Search bot | `constraints(observation)`: what the public record rules out (voids, cards that must be somewhere); a generic default derives it from the legal-move function | the world sampler (`cardbot/sampler.h`), rollouts on shared worlds, candidate racing with paired statistics, the exact endgame (alpha-beta over the `Game`), world weighting by play likelihood | `SearchAgent` |
| 4. DMC networks | `encode(observation)`: state, history and action features, and their sizes | the batched self-play environment over the `Game` (C++, pybind11 template), the Q-GRU and belief models, the DMC trainer with replay, export to the C++ format, the C++ forward passes (`cardbot/nn.h`) | Q and belief nets |
| 5. Bidding model | `encodeBid(observation)` | the counterfactual bid trainer: every legal bid tried on the same deal, finished by the current players; the bid net and its export | bid net |
| 6. Refit the DMC networks | — | the DMC trainer again, on the deals the bid net produces (warm start) | refit Q and belief nets |
| 7. Fast networks | — | distillation of the DMC policy into the small fast net (supervised on its choices) | fast net |
| 8. Fast networks in search | — | play-outs with the fast net, the Q blend and its taper, belief-weighted worlds | the strong `SearchAgent` |

Then evaluation at every step: duplicate deals with mirrored seats, paired
95% intervals, matches to a target score (`cardbot.eval`), and for an app the
C interface, the JNI bridge and the Kotlin platform (`platform/`).

## Everything stays customizable

Every step has a default that works for a new game, and every default can be
replaced by the game's own code, at no cost in speed (templates and policy
classes in C++, not virtual calls in the inner loops; plain functions and
classes to subclass in Python). Nothing is hard-wired that a strong bot may
need to tune:

- **Rules and scoring**: anything a trick-taking game does, through the
  `Game` interface: auctions, declarations, multipliers, held-over points,
  match targets. The framework only calls the interface.
- **Inference and sampling**: a game's own constraints (e.g. pinned cards, where
  announced combinations sit), its own extra sampler state, its own world
  weights (belief net, play likelihood, auction likelihood), or its own
  sampler altogether.
- **Search**: every option open (worlds, racing thresholds, exact-endgame
  depth, Q blend and taper, overrule margin), and the play-out policy, the
  leaf evaluation and the move ordering replaceable; the game can also skip
  the search and use its own.
- **Networks**: the game's own encoders and feature sizes, its own models
  (any PyTorch module that exports to the C++ format, or a custom forward
  pass in C++), and per-contract specialists (as Belot's suit / no-trumps /
  all-trumps nets).
- **Training**: rewards and their scaling (e.g. by the contract's multiplier),
  exploration, replay, learning rates and schedules, warm starts, opponent
  mixes, the bidder used during card-play training, hooks at every stage; the
  trainers are building blocks to compose, not a fixed script.
- **Evaluation and app**: the duel modes and the match rules; in the app,
  the game's own calls, screens, words and look on the shared platform.

A game adopts only the parts it wants: Manille and the two Belotes keep their
tuned engines and take shared pieces one at a time.

## The game interface (C++)

A plain type, not a base class (`cardbot/game.h`): the search copies it
freely and nothing costs a virtual call.

```cpp
#include "cardbot/game.h"

struct MyGame {                       // copyable
    using Action = int;               // cards 0..N-1, then bids, declarations ...
    static constexpr int kPlayers = 4;
    int toMove() const;
    bool finished() const;
    std::vector<Action> legalActions() const;
    void apply(Action a);
    double payoff(int seat) const;    // when finished, e.g. the seat's team margin
    int team(int seat) const;
    // the position with what `observer` cannot see redrawn consistently with what it knows
    // (cardbot/sampler.h's HiddenDeal does the drawing)
    MyGame determinize(int observer, std::mt19937_64& rng) const;
};
```

With that alone: `randomAgent`, `ismctsAgent`, `rolloutAgent` (flat
determinized play-out search) and `duel` (`cardbot/agents.h`, `eval.h`).
For training, an Encoder (`cardbot/env.h`) writes the networks' inputs:

```cpp
struct MyEncoder {
    static constexpr int kStateDim, kPlyDim, kMaxPlies, kActionDim, kMaxLegal;
    static constexpr int kPolicyDim;                                  // optional: fast-policy slots
    MyGame newGame(std::mt19937_64&) const;
    bool learns(const MyGame&) const;                                 // decisions the trainer makes
    MyGame::Action autoAction(const MyGame&, std::mt19937_64&) const; // the others
    int encode(const MyGame&, int seat, float* state, float* history) const;
    void encodeAction(const MyGame&, int seat, MyGame::Action, float* out) const;
    int actionSlot(const MyGame&, int seat, MyGame::Action) const;    // optional, with kPolicyDim
    float reward(const MyGame&, int team) const;
};

// with an auction (steps 5 and 6), cardbot/bidding.h:
struct MyBidEncoder {
    static constexpr int kFeatDim, kHistDim, kMaxHist, kNumBids;
    MyGame newGame(std::mt19937_64&) const;
    bool isBid(const MyGame&) const;                                  // a decision of the auction
    int bidSlot(const MyGame&, int seat, MyGame::Action) const;
    int encode(const MyGame&, int seat, float* feat, float* hist) const;
    float reward(const MyGame&, int team) const;
    MyGame::Action baselineBid(const MyGame&, std::mt19937_64&) const; // step 2
};

PYBIND11_MODULE(mygame_cpp, m) {      // cardbot/python.h
    cardbot::bindBatchEnv<MyGame, MyEncoder, MyBidEncoder>(m, "BatchEnv");
    cardbot::bindBidEnv<MyGame, MyBidEncoder, MyEncoder>(m, "BidEnv");
    cardbot::bindQGru(m);
    cardbot::bindFastPolicy(m);
    cardbot::bindBidModel(m);
}
```

The bid environment's table (which bidder each seat uses on each deal), its
extra bidders and its play agent are set in C++, so a game can train against
mixed partners or fixed opponents as Belot does.

A game with its own search builds it from `cardbot/search.h` (world draws
with the game's weights, candidate racing, the Q blend) as Belot does.

## Python

```python
import mygame_cpp
from cardbot.models import QGRU, FastPolicy, BidNet
from cardbot.train import dmc, distill, bidding

env = mygame_cpp.BatchEnv(2048, seed=1, threads=4)    # auctions by the baseline bidder
q = QGRU(env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM)
dmc.train(env, q, dmc.DMCConfig(episodes=5_000_000))               # step 4
fast = FastPolicy(env.STATE_DIM, env.POLICY_DIM)
distill.train(env, q, fast, distill.DistillConfig(rounds=50))      # a first fast net
fast.export_bin("fast.bin", tag=b"MYP1")

benv = mygame_cpp.BidEnv(threads=4)
benv.use_play_policy("fast.bin", "MYP1")                          # cards by the fast net
bid = BidNet(benv.FEAT_DIM, benv.HIST_DIM, benv.NUM_BIDS)
bid = bidding.train(benv, bid, bidding.BidConfig(iters=200, out="bid.bin", tag=b"MYB1"))  # step 5
print(bidding.evaluate(benv))                                       # against the baseline bidder

env.use_bid_model("bid.bin", "MYB1")                               # step 6: refit
dmc.train(env, q, dmc.DMCConfig(episodes=1_000_000, lr=1e-4))
fast = FastPolicy(env.STATE_DIM, env.POLICY_DIM)
distill.train(env, q, fast, distill.DistillConfig(rounds=50))      # step 7
fast.export_bin("fast.bin", tag=b"MYP1")
# step 8, C++: rolloutAgent<MyGame>(fastPolicyAgent<MyGame, MyEncoder>(model)), or the game's own search
```

The trainers carry the recipes of the Manille and Belote bots (the DMC
replay window, history-length buckets, BF16, AdamW and cosine schedules; the
streaming, EMA bid trainer), and each game-specific part is a hook: DMC is a
class (`dmc.DMCTrainer`) whose `env_flags`, `augment`, `loss`, `after_round`
and `on_round_end` a game overrides (Belot: its suit permutations, its
card-ownership loss, its belief net, its evaluation and checkpoints);
distillation and bidding take functions (collect, augment, generate, predict,
export, loss). Bulgarian Belot trains through them (`python/bg` in
belote-bot), with results identical to its own earlier trainers.

## Order of work

The Bulgarian Belot pipeline already runs all eight steps for one game, so
it is the starting point: each step is generalized from it and Belot is
moved onto the general version, checked for identical results (decisions
and hint values bit for bit in C++; the same training curves for a
fixed-seed run in Python).

1. Network kernels: `cardbot/nn.h` (done, bit-identical).
2. World sampler: `cardbot/sampler.h` (done, bit-identical on Belot).
3. The `Game` interface, `randomAgent`, `ismctsAgent`, `rolloutAgent`, the
   search building blocks (`cardbot/search.h`, bit-identical on Belot's
   search), the exact endgame (`cardbot/endgame.h`: alpha-beta with the
   game's table key, move ordering and equivalent-move hooks, and a play-out
   leaf for it) and duels (done; Belot plays through it).
4. The Python package: environment template, Q-GRU, DMC, fast policy and its
   distillation with the C++ policy agent, counterfactual bidding
   (`cardbot/bidding.h`, `train/bidding.py`) and the refit with the learned
   bidder (done, tested on toy games with and without an auction). Belot's
   trainers become thin adapters between its training runs.
5. A small example game (Whist-like) written from scratch against the
   interface, to prove the pipeline end to end for a game the framework has
   never seen, and the "new game" guide.
6. Belgian Belote and Manille onto it where it pays off.

Training runs import the game repositories' code, so their switch-overs
happen between runs.

## Open source

Apache-2.0, like manil-bot. Third-party: ismcsolver (MIT). Benchmarks for the
README: Bulgarian Belot against Kostov's bots (54.35% ± 1.74 pp of 2000 games
against his Master), Manille's ladder.
