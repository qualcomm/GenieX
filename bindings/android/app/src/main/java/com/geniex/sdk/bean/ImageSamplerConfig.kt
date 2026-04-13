package com.geniex.sdk.bean

data class ImageSamplerConfig(
    /* Sampling method: "ddim", "ddpm", etc. */
    val method: String? = "ddim",
    /* Number of denoising steps */
    val steps: Int? = 20,
    /* Classifier-free guidance scale */
    val guidance_scale: Float? = 7.5f,
    /* DDIM eta parameter */
    val eta: Float? = 0f,
    /* Random seed (-1 for random) */
    val seed: Int? = 2
)
