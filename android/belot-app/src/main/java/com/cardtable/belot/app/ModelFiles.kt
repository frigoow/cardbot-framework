package com.cardtable.belot.app

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/**
 * Copies the bundled nets out of the APK assets into the app's private
 * storage, where the C++ engine reads them. The copies are refreshed when the
 * app was installed or updated after they were made (re-exported nets ship
 * under the same names).
 */
object ModelFiles {
    fun installAll(context: Context): File {
        val dir = File(context.filesDir, "models").apply { mkdirs() }
        val installed = runCatching {
            context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        }.getOrDefault(Long.MAX_VALUE)
        for (name in context.assets.list("models").orEmpty()) {
            if (!name.endsWith(".bin")) continue
            val target = File(dir, name)
            if (target.isFile && target.lastModified() >= installed) continue
            val tmp = File(dir, "$name.tmp")
            context.assets.open("models/$name").use { input -> FileOutputStream(tmp).use(input::copyTo) }
            if (!tmp.renameTo(target)) error("could not install $name")
        }
        return dir
    }
}
