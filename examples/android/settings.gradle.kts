enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")

pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
// CI override: when GENIEX_AAR_REPO names a local Maven repository, resolve
// com.qualcomm.qti:geniex-android from there instead of Maven Central. CI
// populates it from the AAR built in the same run (see _build-android-apk.yml),
// so release/PR builds exercise the exact artifact being shipped *before* it is
// published to Maven Central — otherwise a release that bumps the example's pin
// to an as-yet-unpublished version fails dependency resolution. It must be a
// real Maven repo (POM with `aar` packaging), not flatDir: AGP's variant-aware
// resolution needs that metadata to accept a versioned coordinate as an AAR.
// Real users never set this and pull the published version from Maven Central.
val geniexAarRepo: String? = System.getenv("GENIEX_AAR_REPO")

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        if (geniexAarRepo != null) {
            // Scoped to the one module so this repo is never consulted for any
            // other dependency.
            maven {
                url = uri(geniexAarRepo)
                content { includeModule("com.qualcomm.qti", "geniex-android") }
            }
        }
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") } // Added JitPack for AndroidAutoSize
//        maven {
//            url = uri("https://raw.githubusercontent.com/GenieXAI/core/main")
//        }
        flatDir {
            dirs("app/libs")
        }
    }
}

rootProject.name = "GenieXDemo"

include(":transform")
include(":app")

