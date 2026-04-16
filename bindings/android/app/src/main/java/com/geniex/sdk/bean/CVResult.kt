package com.geniex.sdk.bean

class CVResult{
    /* Output image paths (caller must free with ml_free) */
    var image_paths: ArrayList<String>? = null
    /* Number of output images */
    var image_count: Int = 0
    /* Class ID (example: ConvNext) */
    var class_id: Int = 0
    /* Confidence score [0.0-1.0] */
    var confidence: Float = 0f
    /* Bounding box (example: YOLO) */
    var bbox: BoundingBox? = null
    var text: String? = null
    /* Feature embedding (example: CLIP embedding) (caller must free with ml_free) */
    var embedding: ArrayList<Float>? = null
    /* Embedding dimension */
    var embedding_dim: Int = 0
}