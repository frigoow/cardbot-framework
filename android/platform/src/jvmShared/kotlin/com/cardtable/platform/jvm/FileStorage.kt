package com.cardtable.platform.jvm

import com.cardtable.platform.Storage
import java.io.File

/** Named files in one directory; writes go to a temp file and are renamed into place. */
class FileStorage(private val dir: File) : Storage {
    init { dir.mkdirs() }

    override fun read(name: String): String? = File(dir, name).takeIf { it.isFile }?.readText()

    override fun write(name: String, text: String) {
        val file = File(dir, name)
        val tmp = File(dir, "$name.tmp")
        tmp.writeText(text)
        if (!tmp.renameTo(file)) { file.delete(); tmp.renameTo(file) }
    }

    override fun delete(name: String) { File(dir, name).delete() }
    override fun exists(name: String): Boolean = File(dir, name).isFile
    override fun rename(from: String, to: String): Boolean = File(dir, from).renameTo(File(dir, to))
}
