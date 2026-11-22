# cardbot-framework

The shared building blocks of our card-game apps (Android, Windows, macOS,
Linux): what every game's app needs that is not the game. Each game keeps its
engine, bots and app in its own repository and includes cardbot-framework as a git
submodule:

| Game | Repository | App |
|---|---|---|
| Manille | [manil-bot](https://github.com/frigoow/manil-bot) | `android/` |
| Bulgarian Belot | belote-bot | `app/` |

## What is in it

| Path | What |
|---|---|
| `platform/` | A Kotlin Multiplatform + Compose module: profiles, saved games and statistics files; English plus the game's language (`tr`, and `tl` for these shared screens); the table themes; cards drawn through each game's `Deck` (its own card numbering); speech bubbles, prompts, score boxes, the profile dialog and theme picker; the felt table and its scenery; the worker thread the bots think on; file storage on the JVMs; desktop paths |
| `cpp/` | The game-independent C++ of the bots (see [docs/DESIGN.md](docs/DESIGN.md)), header-only: `game.h` (the interface a game implements), `agents.h` (random and ISMCTS baselines, the latter through the bundled ismcsolver), `eval.h` (duplicate-deal duels with paired 95% intervals), `sampler.h` (the world sampler's counting core), `search.h` (world drawing, candidate racing, Q blend, overrule test) and `nn.h` (network building blocks). A game's CMake: `add_subdirectory(cardbot-framework/cpp cardbot)` and link `cardbot` |
| `build-logic/` | The desktop apps' native build: `CMakeJniLibrary` (a game engine's JNI library, built with CMake for the OS at hand), `findCmake`, `findJniInclude` |

No game rules live here.

## Using it in a game

```bash
git submodule add https://github.com/frigoow/cardbot-framework.git cardbot-framework
```

In the app's Gradle build (say `app/`, beside the submodule):

```kotlin
// settings.gradle.kts
include(":platform")
project(":platform").projectDir = file("../cardbot-framework/platform")

// buildSrc/build.gradle.kts
plugins { `kotlin-dsl` }
repositories { mavenCentral() }
sourceSets { main { kotlin.srcDir("../../cardbot-framework/build-logic") } }
```

and declare the same plugins as `build.gradle.kts` here. The game module
depends on `project(":platform")`, provides its `Deck`, and draws its own
calls and score with the shared widgets. Belot's app (belote-bot `app/`) is
the full example: a C interface over the engine (per-seat views as JSON,
actions as ints, so an online table could reuse it), a JNI bridge, the
table's controller and screens, and an Android and a desktop shell.

A change here reaches a game when its repository moves the submodule:
`git -C cardbot-framework pull origin main` and commit the new pointer.

## Build

```bash
./build.sh :platform:compileKotlinDesktop    # build.sh finds a JDK, like the apps' own
```
