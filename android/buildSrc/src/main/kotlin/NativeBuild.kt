import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.Project
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import java.io.File
import javax.inject.Inject

// The desktop apps' native side: a game engine's JNI library, built with
// CMake for the OS that runs the build (MSVC, Apple clang, GCC).

/**
 * Configures [sources] with CMake (Ninja when it sits next to cmake), passing
 * [definitions] as -D flags plus JNI_INCLUDE_DIRS, and builds [target]. The
 * CMake project puts the library in the directory it is given (here
 * [outputDir], through the definitions).
 */
abstract class CMakeJniLibrary @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:InputDirectory @get:PathSensitive(PathSensitivity.RELATIVE) abstract val sources: DirectoryProperty
    /** More sources the build reads (an engine outside [sources]); may repeat [sources]. */
    @get:InputDirectory @get:PathSensitive(PathSensitivity.RELATIVE) abstract val extraSources: DirectoryProperty
    @get:Internal abstract val workDir: DirectoryProperty
    @get:OutputDirectory abstract val outputDir: DirectoryProperty
    @get:Input abstract val cmake: Property<String>
    @get:Input abstract val jniInclude: ListProperty<String>
    @get:Input abstract val target: Property<String>
    @get:Input abstract val definitions: ListProperty<String>

    @TaskAction fun build() {
        val work = workDir.get().asFile
        val ninja = File(cmake.get()).parentFile?.resolve("ninja")?.takeIf { it.canExecute() }
        exec.exec {
            commandLine(buildList {
                add(cmake.get()); add("-S"); add(sources.get().asFile.path); add("-B"); add(work.path)
                if (ninja != null) { add("-G"); add("Ninja"); add("-DCMAKE_MAKE_PROGRAM=${ninja.path}") }
                add("-DCMAKE_BUILD_TYPE=Release")
                definitions.get().forEach { add("-D$it") }
                add("-DJNI_INCLUDE_DIRS=${jniInclude.get().joinToString(";")}")
            })
        }
        exec.exec { commandLine(cmake.get(), "--build", work.path, "--config", "Release", "--target", target.get(), "--parallel") }
    }
}

private val isWindows = System.getProperty("os.name").lowercase().contains("win")

/** cmake: -Pmanille.cmake, else PATH, else the newest one in the Android SDK. */
fun Project.findCmake(): String {
    (findProperty("manille.cmake") as String?)?.let { return it }
    val exe = if (isWindows) "cmake.exe" else "cmake"
    System.getenv("PATH").orEmpty().split(File.pathSeparator)
        .map { File(it, exe) }.firstOrNull { it.canExecute() }?.let { return it.path }
    val sdk = System.getenv("ANDROID_HOME") ?: "${System.getProperty("user.home")}/Android/Sdk"
    return File(sdk, "cmake").listFiles().orEmpty().sortedByDescending { it.name }
        .map { File(it, "bin/$exe") }.firstOrNull { it.canExecute() }?.path
        ?: throw GradleException("cmake not found: install it, or pass -Pmanille.cmake=/path/to/cmake")
}

/**
 * jni.h: from a full JDK (the one running Gradle, JAVA_HOME, the usual
 * install places). A runtime-only JDK such as Android Studio's has no
 * headers; then the NDK's self-contained jni.h, which has the same ABI.
 */
fun Project.findJniInclude(): List<String> {
    val homes = buildList {
        add(File(System.getProperty("java.home")))
        System.getenv("JAVA_HOME")?.let { add(File(it)) }
        addAll(File(System.getProperty("user.home"), ".jdks").listFiles().orEmpty())
        addAll(File("/usr/lib/jvm").listFiles().orEmpty())
        addAll(File("/Library/Java/JavaVirtualMachines").listFiles().orEmpty().map { File(it, "Contents/Home") })
    }
    homes.firstOrNull { File(it, "include/jni.h").isFile }?.let { home ->
        val include = File(home, "include")
        return listOf(include.path) + include.listFiles().orEmpty().filter { it.isDirectory }.map { it.path }
    }
    val sdk = System.getenv("ANDROID_HOME") ?: "${System.getProperty("user.home")}/Android/Sdk"
    val ndkJni = File(sdk, "ndk").listFiles().orEmpty().sortedByDescending { it.name }
        .flatMap { File(it, "toolchains/llvm/prebuilt").listFiles().orEmpty().toList() }
        .map { File(it, "sysroot/usr/include/jni.h") }.firstOrNull { it.isFile }
        ?: throw GradleException("No jni.h: install a full JDK (e.g. openjdk-21-jdk-headless) or set JAVA_HOME to one.")
    // only jni.h itself: the rest of the NDK sysroot is Android's libc
    val dir = layout.buildDirectory.dir("jni-include").get().asFile
    dir.mkdirs(); ndkJni.copyTo(File(dir, "jni.h"), overwrite = true)
    return listOf(dir.path)
}
