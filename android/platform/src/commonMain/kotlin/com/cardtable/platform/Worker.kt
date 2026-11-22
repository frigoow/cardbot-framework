package com.cardtable.platform

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex

/**
 * The one thread the bots and hints think on, away from the screen. A job
 * holds [lock] for its whole run, so closing the game can wait for the job
 * in flight before it frees what the job uses.
 */
class Worker {
    val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
    val lock = Mutex()
}
