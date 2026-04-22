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
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") } // Added JitPack for AndroidAutoSize
//        maven {
//            url = uri("https://raw.githubusercontent.com/GeniexAI/core/main")
//        }
        flatDir {
            dirs("app/libs")
        }
    }
}

rootProject.name = "GeniexDemo"

include(":transform")
include(":app")

// Consume the Android bindings library module directly from this repo
// (replaces the previous ai.geniex:core Maven dependency).
include(":geniex-bindings")
project(":geniex-bindings").projectDir = file("../../bindings/android/app")

