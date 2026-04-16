import com.android.build.api.variant.LibraryVariant
import com.android.build.gradle.api.BaseVariant

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

apply {
    from("update.gradle")
}

android {
    namespace = "com.geniex.sdk"
    compileSdk = 35
    ndkVersion = "29.0.13846066"

    defaultConfig {
        minSdk = 27

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf(
                    "-DGENIEX_DL=OFF",
//                    "-DGENIEX_ANDROID=ON"
                )
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
    buildFeatures {
//        viewBinding = true
    }
}

afterEvaluate {
    android.libraryVariants.forEach {
        registerCopyBridgeLibsTask(it)
    }
}

fun registerCopyBridgeLibsTask(variant: BaseVariant) {
    // Use SDK install directory structure
    val sdkInstallDir = File(projectDir,"./../../../build-android/install")
    println("SDK install directory: $sdkInstallDir")
    require(sdkInstallDir.exists()) {
        "SDK install directory not found at: ${sdkInstallDir.absolutePath}.\n" +
        "Please run: cd sdk && cmake --install build-arm64-android-snapdragon-release --prefix ../build-android/install"
    }

    val sdkLibDir = File(sdkInstallDir, "lib")
    val cap = variant.name.replaceFirstChar { it.uppercase() }

    val jniOutDir = File(projectDir, "src/main/jniLibs/arm64-v8a")
    if (!jniOutDir.exists()) {
        jniOutDir.mkdirs()
    }
    println("Copying SDK libraries to ${jniOutDir.absolutePath}")

    val copyTask = tasks.register<Copy>("copyBridgeLibs$cap") {
        // Copy main SDK library
        from(sdkLibDir) {
            include("libgeniex.so")
        }
        // Copy llama_cpp plugin and dependencies
        from(File(sdkLibDir, "llama_cpp")) {
            include("*.so")
            exclude("*.a")  // Exclude static libraries
        }
        // Copy external libs if they exist
        from(File(projectDir, "extLibs/arm64-v8a")) {
            include("*.so")
        }

        into(jniOutDir)
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE

        eachFile {
            if (name == "libgeniex_plugin.so") {
                // Rename plugin to include backend name
                path = "libgeniex_plugin_llama_cpp.so"
                println("Found geniex plugin, rename to: $path")
            } else {
                path = name
            }
        }
    }

    // Copy HTP assets for QAIRT plugin (if exists)
    val htpAssetsDir = File(projectDir, "src/main/assets/qairt")
    val qnnLibDir = File(sdkLibDir, "qairt")
    val copyHtpTask = tasks.register<Copy>("copyHtpAssets$cap") {
        from(qnnLibDir) {
            include("htp-files*/**")
        }
        into(htpAssetsDir)
        includeEmptyDirs = false
    }

    listOf(
//        "merge${cap}NativeLibs",
//        "merge${cap}JniLibs",
//        "pre${cap}Build"
        "preBuild"
    ).forEach { n ->
        tasks.matching { it.name == n }.configureEach {
            dependsOn(copyTask)
            dependsOn(copyHtpTask)
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
//    implementation(libs.androidx.appcompat)
//    implementation(libs.material)
//    implementation(libs.androidx.constraintlayout)
//    testImplementation(libs.junit)
//    androidTestImplementation(libs.androidx.junit)
//    androidTestImplementation(libs.androidx.espresso.core)
    implementation(libs.kotlinx.coroutines.android)
}
