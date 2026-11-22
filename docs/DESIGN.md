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

## The game interface (C++)

```cpp
#include "cardbot/game.h"

struct MyGame {                       // copyable: the search copies it freely
    using Action = int;               // cards 0..N-1, then bids, declarations ...
    int toMove() const;
    bool finished() const;
    std::vector<Action> legalActions() const;
    void apply(Action a);
    double score(int team) const;     // when finished

    using Observation = ...;          // what one seat knows
    Observation observe(int seat) const;
    // a full game consistent with an observation, from hidden hands the sampler drew
    static MyGame fromWorld(const Observation&, const cardbot::Hands&);
};
// optional, for steps 3 to 5:
cardbot::Constraints constraints(const MyGame::Observation&);
void encode(const MyGame::Observation&, cardbot::Features&);
```

## Python

```python
import cardbot, mygame_cpp                      # pybind11 module from cardbot's template
env = cardbot.Env(mygame_cpp, num_envs=2048)    # batched self-play over the C++ game
cardbot.train.dmc(env, out="checkpoints/q", episodes=5_000_000)          # step 4
cardbot.train.bidding(env, play="checkpoints/q", out="checkpoints/bid")  # step 5
cardbot.train.dmc(env, init="checkpoints/q", bid="checkpoints/bid", out="checkpoints/q2")  # 6
cardbot.train.distill(env, teacher="checkpoints/q2", out="checkpoints/fast")             # 7
print(cardbot.eval.duel(env, a="search:fast=checkpoints/fast", b="ismcts:1000", deals=2000))  # 8
```

## Order of work

The Bulgarian Belot pipeline already runs all eight steps for one game, so
it is the starting point: each step is generalized from it and Belot is
moved onto the general version, checked for identical results (decisions
and hint values bit for bit in C++; the same training curves for a
fixed-seed run in Python).

1. Network kernels: `cardbot/nn.h` (done, bit-identical).
2. World sampler: `cardbot/sampler.h` (core done, bit-identical on Belot).
3. The `Game` interface, `RandomAgent`, `IsmctsAgent`, the generic search
   (`SearchAgent`) and endgame, with Belot as the first implementation.
4. The Python package: environment template, models, DMC, bidding, refit,
   distillation, export, evaluation; Belot's trainers become thin adapters.
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
