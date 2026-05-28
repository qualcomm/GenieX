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
// CI override: when -PgeniexAarDir=<dir> is passed, resolve
// com.qualcomm.qti:geniex-android from a locally-built AAR (named
// geniex-android.aar) in that directory instead of Maven Central. Lets
// release/PR CI build the demo against the exact AAR being shipped, before it
// is published to Maven Central — without that, a release that bumps the pin
// to an as-yet-unpublished version would fail to resolve. Real users never set
// this property and pull the published version from Maven Central as usual.
val geniexAarDir: String? = startParameter.projectProperties["geniexAarDir"]

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        if (geniexAarDir != null) {
            // Scoped to the one module so this repo is never consulted for any
            // other dependency (no flat-dir resolution noise). flatDir carries
            // no metadata, but the published geniex-android POM declares no
            // dependencies either, so the resulting classpath is identical.
            flatDir {
                dirs(geniexAarDir)
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

