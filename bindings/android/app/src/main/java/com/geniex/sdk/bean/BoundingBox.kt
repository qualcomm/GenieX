package com.geniex.sdk.bean

data class BoundingBox(
    val x: Float,      /* X coordinate (normalized or pixel, depends on model) */
    val y: Float,      /* Y coordinate (normalized or pixel, depends on model) */
    val width: Float,  /* Width */
    val height: Float, /* Height */
)