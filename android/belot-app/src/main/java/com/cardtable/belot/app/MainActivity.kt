package com.cardtable.belot.app

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.cardtable.belot.BULGARIAN_BOT_NAMES
import com.cardtable.belot.BelotPlatform
import com.cardtable.belot.BelotRoot
import com.cardtable.belot.jvm.JniBelotEngine
import com.cardtable.platform.AppConfig
import com.cardtable.platform.jvm.FileStorage

/** The Android shell: the engine and its nets (out of the APK assets), then the Belot screens. */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        CrashReport.install(this)
        if (savedInstanceState == null) CrashReport.showIfAny(this)
        AppConfig.botNames = BULGARIAN_BOT_NAMES
        val log = { m: String -> Log.i("belot", m); Unit }
        val models = ModelFiles.installAll(applicationContext)
        val platform = BelotPlatform(
            engine = JniBelotEngine(models.path, log),
            storage = FileStorage(applicationContext.filesDir),
            log = log,
            epochMillis = System::currentTimeMillis,
            systemLanguage = java.util.Locale.getDefault().language,
        )
        setContent { BelotRoot(platform) }
    }
}
