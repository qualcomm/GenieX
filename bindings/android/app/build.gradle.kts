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
    val bridgeOutDir = File(projectDir,"./../../../build-android/out/")
    println("bridgeOutDir: $bridgeOutDir")
    require(bridgeOutDir.exists()) { "SDK build-android not found at: ${bridgeOutDir.absolutePath}. Run scripts/build_android.sh." }

    val cap = variant.name.replaceFirstChar { it.uppercase() }

    val jniOutDir = File(projectDir, "src/main/jniLibs/arm64-v8a")
    if (!jniOutDir.exists()) {
        jniOutDir.mkdirs()
    }
    print("copy geniex sdk libraries to ${jniOutDir.absolutePath}\n")

    val copyTask = tasks.register<Copy>("copyBridgeLibs$cap") {
        from(File(projectDir, "extLibs/arm64-v8a"))
        from(bridgeOutDir)
        into(jniOutDir)

        include("**/*.so")
        exclude("**/htp-files*/**")
        includeEmptyDirs = false

        duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        eachFile {
            if (name == "libgeniex_plugin.so") {
                val dir = file.parentFile!!.name!!
                path = "libgeniex_plugin_${dir}.so"
                println("Found geniex bridge lib in dir: $dir, rename to: $path")
            } else {
                path = name
            }
        }
    }

    val htpAssetsDir = File(projectDir, "src/main/assets/npu")
    val npuBuildDir = File(bridgeOutDir, "npu")
    val copyHtpTask = tasks.register<Copy>("copyHtpAssets$cap") {
        from(npuBuildDir) {
            include("htp-files/**")
            include("htp-files-v81/**")
            include("htp-files-v85/**")
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
