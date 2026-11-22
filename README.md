# cardtable

The apps for our card-game bots: one app per game (Android, Windows, macOS,
Linux), built from shared building blocks. Each game's rules engine and bots
live in that game's own research repository and come in here as a git
submodule, pinned to a commit whose bots have been evaluated.

| Game | Engine (submodule) | App | Releases |
|---|---|---|---|
| Manille | `engines/manille` → [manil-bot](https://github.com/frigoow/manil-bot) | `android/` | [manil-bot releases](https://github.com/frigoow/manil-bot/releases) |
| Bulgarian Belot | `engines/belote` → belote-bot (`cpp/bg`) | to come | to come |

## Status

Step 1 of the plan below is done: the Manille app of manil-bot v1.2.3
(`7b0c72d`) moved here unchanged, with its build pointed at
`engines/manille/cpp`. It builds the same APK and desktop packages.

## Plan

1. **Move Manille's app here unchanged** (done). It keeps its `applicationId` and
   signing key, so new versions still install over the old one.
2. **Split shared from game-specific.** `kotlin/platform` holds the table widgets,
   cards, themes, i18n, profiles, statistics, saves, settings and native
   loading. `kotlin/games/<game>` holds the bidding panel, contract and score
   display, the game's words and its model files. There is one app module per
   game.
3. **One C interface per game engine** (`cg_new`, `cg_view(seat)` as JSON,
   `cg_legal`, `cg_apply`, `cg_bot`, `cg_hint`, `cg_save`/`cg_load`). Actions are
   ints with a game-specific meaning, and a view never holds another seat's
   cards. With one interface, a single JNI bridge and a generic session serve
   every game. Online play can later reuse the same per-seat views, but
   multiplayer is not planned yet.
4. **Bulgarian Belot:** `cpp/bg/capi.cpp` in belote-bot, the `games/belot`
   UI, and a separate app. The bots' networks there run in plain C++, so it
   needs no ONNX Runtime.

## Build

```bash
git clone --recurse-submodules <this repo>
scripts/fetch_models.sh                 # Manille's networks (not in git)
cd android
./build.sh assembleDebug                # APK
./gradlew :desktop:run                  # desktop
```

See [android/README.md](android/README.md) for the toolchain (SDK, NDK, JDK).

## Releases

Push the tag `manille-vX.Y.Z`. CI builds the signed APK and the three desktop
packages and publishes them as release `vX.Y.Z` of manil-bot, at the engine
commit they were built from, so the existing download links keep working.
This needs the secret `MANIL_BOT_RELEASE_TOKEN`, a fine-grained token with
write access to manil-bot's contents. It also needs the existing `MANILLE_KEYSTORE_*`
secrets, copied to this repository.
