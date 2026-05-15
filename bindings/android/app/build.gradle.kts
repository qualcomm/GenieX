plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.geniex.sdk"
    compileSdk = 35
    ndkVersion = "29.0.14206865"

    defaultConfig {
        minSdk = 27

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    packaging {
        jniLibs.useLegacyPackaging = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// Copy the prebuilt geniex libraries from sdk/pkg-geniex so they get packaged
// into jniLibs (and therefore into the final APK of any consumer).
val pkgGeniexDir = file("$projectDir/../../../sdk/pkg-geniex")
val jniOutDir = file("$projectDir/src/main/jniLibs/arm64-v8a")

val copyBridgeLibs = tasks.register<Copy>("copyBridgeLibs") {
    require(pkgGeniexDir.exists()) {
        "SDK package not found at ${pkgGeniexDir.absolutePath}"
    }
    val libDir = File(pkgGeniexDir, "lib")
    from(libDir) { include("libgeniex.so") }
    from(File(libDir, "llama_cpp")) {
        include("*.so")
        exclude("*.a")
        // The SDK package ships multiple libgeniex_plugin.so variants, one per backend.
        // They must be renamed to coexist in the flat APK jniLibs directory.
        rename("libgeniex_plugin\\.so", "libgeniex_plugin_llama_cpp.so")
    }
    from(File(libDir, "qairt")) {
        // Pull every top-level .so so new transitive deps in libgeniex_plugin.so
        // (e.g. libgeniex_vlm.so, libgeniex-proc.so) get packaged automatically.
        // htp-files/ is excluded — see comment below.
        include("*.so")
        exclude("htp-files/**")
        rename("libgeniex_plugin\\.so", "libgeniex_plugin_qairt.so")
    }
    // Do NOT copy from qairt/htp-files/: on Windows CLI packages it ships
    // Hexagon DSP (32-bit) binaries whose basenames collide with the
    // Android CPU-side libs (libQnnSystem.so, libQnnSaver.so), and Android's
    // loader then rejects them with "is 32-bit instead of 64-bit" when
    // QAIRT dlopens them.
    //
    // Instead, pull from the qairt submodule's Android third-party dir
    // which ships both the ARM64 CPU libs and DSP skels under
    // non-colliding names:
    //   - CPU ARM64: libQnnSystem.so, libQnnHtp*.so (dlopen'd by us)
    //   - DSP:      libQnnHtpV??.so, libQnnHtpV??Skel.so, libCalculator_skel.so
    //               (loaded by FastRPC on the Hexagon side; named so they
    //               never collide with CPU libs, so Android happily
    //               ships them alongside the ARM64 ones without trying
    //               to dlopen them in the main process).
    from(File(projectDir, "../../../third-party/geniex-qairt/third-party/android")) {
        include("*.so")
    }
    from(File(projectDir, "extLibs/arm64-v8a")) { include("*.so") }
    into(jniOutDir)
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

tasks.matching { it.name == "preBuild" }.configureEach {
    dependsOn(copyBridgeLibs)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
}
