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
                arguments += listOf("-DGENIEX_DL=OFF")
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
val htpAssetsDir = file("$projectDir/src/main/assets/qairt")

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
        rename("libgeniex_plugin\\.so", "libgeniex_plugin_cpu_gpu.so")
    }
    from(File(libDir, "qairt")) {
        include("libgeniex_core.so", "libgeniex_plugin.so")
        rename("libgeniex_plugin\\.so", "libgeniex_plugin_npu.so")
    }
    from(File(libDir, "qairt/htp-files")) {
        include("*.so")
    }
    from(File(projectDir, "extLibs/arm64-v8a")) { include("*.so") }
    into(jniOutDir)
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

// Non-.so artifacts (e.g. libqnnhtpv81.cat) cannot live under jni/, so they
// continue to ship as assets for consumers to extract at runtime.
val copyHtpAssets = tasks.register<Copy>("copyHtpAssets") {
    from(File(pkgGeniexDir, "lib/qairt/htp-files")) {
        exclude("*.so")
    }
    into(File(htpAssetsDir, "htp-files"))
    includeEmptyDirs = false
}

tasks.matching { it.name == "preBuild" }.configureEach {
    dependsOn(copyBridgeLibs)
    dependsOn(copyHtpAssets)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
}
