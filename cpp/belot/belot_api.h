#pragma once
// Bulgarian Belot for the apps: the referee (engines/belote, bg::Round and
// bg::GameScore), the bots and the hints behind a plain C interface, the same
// on Android and the desktop (jni_belot.cpp) and, later, on a server.
//
// A game is a sequence of deals to 151 points (or one deal). Seats are the
// engine's: S=0, E=1, N=2, W=3 in playing order (counter-clockwise), teams
// S/N = 0 and E/W = 1. A seat only ever sees its own view (belot_view):
// its cards, the public record, and never another seat's hand.
//
// Actions are ints:
//   0..31        play that card (belot is claimed whenever it may be)
//   32 + b       bid b: 0 pass, 1 clubs, 2 diamonds, 3 hearts, 4 spades,
//                5 no trumps, 6 all trumps, 7 double, 8 redouble
//   64 + mask    declare the offered combinations whose index is set in mask
//                (0: declare none)
// Cards are suit * 8 + rank: suits clubs, diamonds, hearts, spades; ranks
// 7 8 9 10 J Q K A.
//
// Every function returns >= 0 on success and < 0 on an error
// (belot_last_error says which). The text functions return the length of the
// text, write it into buf only when it fits (cap includes the NUL), and may be
// called with buf = NULL to learn the length.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BelotGame BelotGame;

enum {
    BELOT_ACTION_BID = 32,
    BELOT_ACTION_DECLARE = 64,
    BELOT_LEVELS = 3,  // bot strength 0 (rule policy, instant) .. 2 (full search)
};

/** The last error on this thread. */
const char* belot_last_error(void);

/**
 * Loads the networks from dir (belot_bid.bin and belot_{suit,nt,at}_{fast,
 * belief,q}.bin). Missing files are skipped: the bots then bid by rule and
 * search without those nets. Call once, before any bot or hint runs.
 * Returns the number of files loaded.
 */
int belot_models_load(const char* dir);
/** Which networks are loaded, as text (diagnostics). */
int belot_models_info(char* buf, int cap);

/** A new game: match = 1 plays to 151, 0 a single deal. The first deal is started. */
BelotGame* belot_new(uint64_t seed, int match);
BelotGame* belot_clone(const BelotGame* game);
void belot_free(BelotGame* game);

/** Starts the next deal once the current one is over (not after the game is). */
int belot_next_deal(BelotGame* game);

/** What seat may see now, as JSON (see belot_api.cpp: viewJson). */
int belot_view(const BelotGame* game, int seat, char* buf, int cap);
/** Another seat's current cards (the "open cards" debug view), as a mask. */
int belot_hand(const BelotGame* game, int seat, int64_t* out_mask);

/** seat makes action; fails if it is not seat's turn or the action is illegal. */
int belot_apply(BelotGame* game, int seat, int action);

/**
 * The bot's action for seat, the seat to move, at strength level. Reads only
 * what seat may see. Blocks while it searches: never call it on a UI thread.
 */
int belot_bot_action(const BelotGame* game, int seat, int level, uint64_t seed, int* out_action);

/**
 * The bot's analysis of seat's position: each legal action's value in game
 * points (for the seat's team) into actions/values (at most cap), and the
 * bot's own pick in out_pick. Returns how many were written: 0 where there is
 * nothing to weigh (declarations).
 */
int belot_hint(const BelotGame* game, int seat, int level, uint64_t seed, int* actions, double* values, int cap,
               int* out_pick);

/** The game as text, to continue it later with belot_load. */
int belot_save(const BelotGame* game, char* buf, int cap);
BelotGame* belot_load(const char* text);

#ifdef __cplusplus
}
#endif
