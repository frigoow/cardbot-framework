# cardtable

The apps for our card-game bots: one app per game (Android, Windows, macOS,
Linux), built from shared building blocks. Each game's rules engine and bots
live in that game's own research repository and come in here as a git
submodule, pinned to a commit whose bots have been evaluated.

| Game | Engine (submodule) | Modules | Releases |
|---|---|---|---|
| Manille | `engines/manille` → [manil-bot](https://github.com/frigoow/manil-bot) (public) | `android/manille`, `app`, `desktop` | [manil-bot releases](https://github.com/frigoow/manil-bot/releases) |
| Bulgarian Belot | `engines/belote` → belote-bot (private), `cpp/belot` | `android/belot`, `belot-app`, `belot-desktop` | [cardtable releases](https://github.com/frigoow/cardtable/releases) |

## Layout

| Path | What |
|---|---|
| `android/platform` | Shared by every app: profiles, saved games and statistics files, English + the game's language (`tr` / `tl`), the table themes, cards drawn through each game's `Deck`, speech bubbles, prompts, the felt table and its scenery, the worker thread |
| `android/buildSrc` | The desktop apps' native build: CMake to a JNI library for the OS at hand |
| `android/manille` | Manille's table, bots' Kotlin side (ONNX Runtime) and screens |
| `android/belot` | Belot's table (`BelotController`: the referee's record shown in order, the bots' turns on the worker, hints, statistics, the saved game) and screens (`BelotUi`) |
| `cpp/belot` | `belot_api.h`: Belot's referee, bots and hints behind a plain C interface (per-seat views as JSON, actions as ints), its JNI bridge and tests |
| `scripts/` | `fetch_models.sh` (Manille's nets, from the models release), `belot_models.sh` (Belot's nets, from a belote-bot checkout) |

A new game brings its engine as a submodule, a C interface like
`cpp/belot/belot_api.h`, a game module like `android/belot` that draws its
own calls and score on the platform's widgets, and two thin shells. The same
per-seat views and integer actions are what an online table would send, so
multiplayer can be added later without changing the engines.

## Bulgarian Belot

The bots are the strongest measured configuration of belote-bot: the
unmasked neural bidder (with doubles), card play by world sampling with the
belief, fast-policy and Q-GRU nets of the three contract kinds, racing, and
an exact endgame. On 2000 games inside Kostov's BelotGameEngine it won
54.35% ± 1.74 pp against his Master bot. In the app this is "Strong" (192
worlds, about 120 ms a card on a desktop CPU without AVX2); "Normal" searches
64 worlds and "Easy" plays the rule policy. The app starts in Bulgarian on a
Bulgarian device and in English elsewhere; both can be switched in the lobby
or the menu. It uses the same table themes as Manille, its café table first.

## Build

```bash
git clone --recurse-submodules <this repo>      # engines/belote needs access to belote-bot
scripts/fetch_models.sh                         # Manille's networks (not in git)
scripts/fetch_belot_models.sh                   # Belot's networks (or scripts/belot_models.sh ../Belote)
cd android
./build.sh assembleDebug                        # both APKs
./gradlew :desktop:run                          # Manille on the desktop
./gradlew :belot-desktop:run                    # Belot on the desktop
./gradlew -Pmanille.skipAndroidApp=true :manille:desktopTest :belot:desktopTest
cmake -B build -S ../cpp/belot && cmake --build build -j && ctest --test-dir build
```

The desktop modules can also render the real app offscreen into PNGs:
`./gradlew :belot-desktop:test -Pbelot.screenshots=/some/dir` (and
`:desktop:test -Pmanille.screenshots=...`). See
[android/README.md](android/README.md) for the toolchain (SDK, NDK, JDK).

## Releases

For now Manille is still released from manil-bot itself, which holds the
signing key. The workflow here is ready for when that moves: push the tag
`manille-vX.Y.Z`. CI builds the signed APK and the three desktop packages and
publishes them as release `vX.Y.Z` of manil-bot, at the engine commit they
were built from, so the existing download links keep working. This needs
the secret `MANIL_BOT_RELEASE_TOKEN`, a fine-grained token with write access
to manil-bot's contents, and the existing `MANILLE_KEYSTORE_*` secrets copied
to this repository.

Belot is released here (this repository's releases), tagged `belot-vX.Y.Z`.
Its networks are the release `belot-models-v1` (`scripts/fetch_belot_models.sh`).
The signing key is `~/.android/belot-release.jks` on the machine that made
v0.1.0, with its passwords in `belot-release.properties` beside it: keep a
backup, because an APK signed with any other key cannot update an installed
Belot. Pushing a `belot-v*` tag builds everything in CI
(`belot-release.yml`, including the Windows and macOS packages) once the
secrets `BELOTE_BOT_DEPLOY_KEY` (a read-only deploy key of belote-bot, which
CI's `belot` job needs as well) and `BELOT_KEYSTORE_B64`,
`BELOT_KEYSTORE_PASSWORD`, `BELOT_KEY_ALIAS` (`belot`) and
`BELOT_KEY_PASSWORD` are set; until then those jobs only warn.
