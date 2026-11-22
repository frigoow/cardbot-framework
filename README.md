# cardtable

The shared building blocks of our card-game apps (Android, Windows, macOS,
Linux): what every game's app needs that is not the game. Each game keeps its
engine, bots and app in its own repository and includes cardtable as a git
submodule:

| Game | Repository | App |
|---|---|---|
| Manille | [manil-bot](https://github.com/frigoow/manil-bot) | `android/` |
| Bulgarian Belot | belote-bot | `app/` |

## What is in it

| Path | What |
|---|---|
| `platform/` | A Kotlin Multiplatform + Compose module: profiles, saved games and statistics files; English plus the game's language (`tr`, and `tl` for these shared screens); the table themes; cards drawn through each game's `Deck` (its own card numbering); speech bubbles, prompts, score boxes, the profile dialog and theme picker; the felt table and its scenery; the worker thread the bots think on; file storage on the JVMs; desktop paths |
| `build-logic/` | The desktop apps' native build: `CMakeJniLibrary` (a game engine's JNI library, built with CMake for the OS at hand), `findCmake`, `findJniInclude` |

No game rules live here.

## Using it in a game

```bash
git submodule add https://github.com/frigoow/cardtable.git cardtable
```

In the app's Gradle build (say `app/`, beside the submodule):

```kotlin
// settings.gradle.kts
include(":platform")
project(":platform").projectDir = file("../cardtable/platform")

// buildSrc/build.gradle.kts
plugins { `kotlin-dsl` }
repositories { mavenCentral() }
sourceSets { main { kotlin.srcDir("../../cardtable/build-logic") } }
```

and declare the same plugins as `build.gradle.kts` here. The game module
depends on `project(":platform")`, provides its `Deck`, and draws its own
calls and score with the shared widgets. Belot's app (belote-bot `app/`) is
the full example: a C interface over the engine (per-seat views as JSON,
actions as ints, so an online table could reuse it), a JNI bridge, the
table's controller and screens, and an Android and a desktop shell.

A change here reaches a game when its repository moves the submodule:
`git -C cardtable pull origin main` and commit the new pointer.

## Build

```bash
./build.sh :platform:compileKotlinDesktop    # build.sh finds a JDK, like the apps' own
```
