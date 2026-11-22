# cardbot-framework: design

The goal: someone with a new trick-taking card game writes its rules, a
feature encoder and a few game-specific policies, and gets the rest for
free: a fast world sampler, a strong search player, self-play training of
its networks, evaluation with confidence intervals, and an app for Android,
Windows, macOS and Linux. Like [RLCard](https://rlcard.org/getting_started.html),
but with the methods that made the Manille and Belote bots strong (exact
world sampling, rollouts with learned policies, an exact endgame, DMC
self-play) instead of only textbook baselines.

Today three games carry their own copy of most of this: Manille (~11k lines
of C++, ~22k of Python experiments), Belgian Belote (~7k C++, ~3k Python) and
Bulgarian Belot (~5.5k C++, ~1.8k Python, ~2k Kotlin app code). The two
Belote trainers are the same program with renamed classes. Roughly half of
each game's code is not about the game.

## Getting started (the target)

```python
import cardbot
import cardbot_belot          # a game package registers itself: cardbot.make("belot")

env = cardbot.make("belot", seed=1)
env.set_agents([
    cardbot.agents.SearchAgent(env, worlds=192),   # the strong C++ player
    cardbot.agents.RandomAgent(env),
    cardbot.agents.SearchAgent(env, worlds=192),
    cardbot.agents.RandomAgent(env),
])
trajectories, payoffs = env.run()

# self-play training of the game's networks, then a duplicate-deal duel
cardbot.train.dmc(env, out="checkpoints/belot", episodes=5_000_000)
print(cardbot.eval.duel(env, a="search:q=checkpoints/belot", b="rule", deals=2000))
```

and from C++, for the search and the apps:

```cpp
#include "cardbot/search.h"
struct Belot : cardbot::TrickGame<Belot> { /* rules, encoder, policies */ };
auto card = cardbot::searchMove<Belot>(publicState, options);
```

## Layers

| Layer | What it holds | A game provides |
|---|---|---|
| `cpp/include/cardbot/core` | Cards and decks (32, 36, 52), card sets, trick resolution with trump and strength tables, the `Game` concept, match scoring (race to N, held-over points) | its rules as a `Game` |
| `cpp/include/cardbot/sampler` | The exact world sampler: deals the hidden cards uniformly or belief-weighted, under per-card constraints (voids, pinned cards, hand sizes), with an optional game-defined state for things like announced runs; exact enumeration late in the deal | the constraints its public record implies |
| `cpp/include/cardbot/nn` | Network kernels and binary formats (`nn.h`, done) and the standard models: fast policy, belief net, Q-GRU, bid net | input sizes |
| `cpp/include/cardbot/search` | Rollouts on shared worlds, candidate racing with paired statistics, the Q-blend, world weighting by play likelihood, ISMCTS | a play-out policy (rules or net) |
| `cpp/include/cardbot/endgame` | Exact alpha-beta over the last tricks with a transposition table | move generation and scoring (from the `Game`) |
| `cpp/include/cardbot/eval` | Duplicate deals, mirrored pairs, 95% intervals, match mode, match equity | nothing |
| `cpp/include/cardbot/capi` | The app interface: per-seat views as JSON, actions as ints, bots, hints, save and restore by replay; a JNI bridge generator | its view fields |
| `python/cardbot` | `make`, the game registry, `Env` (RLCard-like: `reset`, `step`, `run`, `set_agents`), a batched environment over the C++ game, agents (random, rule, search, network), the models in PyTorch, trainers (DMC self-play, belief, distillation, counterfactual bidding), export to the C++ formats, evaluation | a pybind11 module from a template |
| `platform/` (Kotlin) | Profiles, statistics, saves, themes, two languages, cards, widgets, the table and its scenery (done); next the generic table controller: turn order, pacing, forced moves, hints, reactions, the saved game; and the shared screens: lobby, statistics, table, hand, last trick, menus | its calls (bidding panel), its deal sheet and its words |
| `templates/new-game` | A game skeleton wired to all of the above | |
| `examples/` | A small complete game, to show a new game fits and to test the framework | |

## How the existing games move onto it

Each step takes one piece out of the games and makes them use it, and is
checked for identical results before it lands:

1. **Network kernels** (done for Bulgarian Belot): `nn.h`; 290 full-strength
   decisions identical bit for bit.
2. **World sampler**: one counting core for Bulgarian Belot's stateless sampler
   and Belgian Belote's run-tracking one; identical worlds for identical seeds.
3. **Search, endgame, evaluation**: the Bulgarian search first, then the Belgian.
4. **Python**: the environment protocol and the trainers; the Bulgarian
   trainer becomes a thin adapter, checked by a fixed-seed run with the same
   curves; then the Belgian.
5. **App**: the table controller and the shared screens out of `BelotController`
   and `BelotUi`; then Manille's onto the same, keeping its look and its data.
6. **Template and example game**, then the documentation: getting started,
   writing a game, training, the app.

Training runs import the game repositories' code, so their switch-overs
happen between runs.

## What stays in a game's repository

Its rules and their tests (and oracles against other engines), its feature
encoders, its rule-based policies and bidding heuristics, its networks and
results, its app's calls, words and look, and its releases.

## Open source

Apache-2.0, like manil-bot. Third-party: ismcsolver (MIT). Benchmarks for the
README: Bulgarian Belot against Kostov's bots (54.35% ± 1.74 pp of 2000 games
against his Master), Manille's ladder.
