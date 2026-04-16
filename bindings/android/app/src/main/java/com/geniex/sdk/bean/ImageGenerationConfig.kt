package com.geniex.sdk.bean

data class ImageGenerationConfig(
    /* Required positive prompts */
    val prompts: Array<String>,
    /* Optional negative prompts */
    val negative_prompts: Array<String>? = null,
    /* Output image height */
    val height: Int? = 512,
    /* Output image width */
    val width: Int? = 512,
    /* Sampling parameters */
    val sampler_config: ImageSamplerConfig? = ImageSamplerConfig(),
    /* Scheduler configuration */
    val scheduler_config: SchedulerConfig? = SchedulerConfig(),
    /* Denoising strength for img2img */
    val strength: Float? = 1.0f
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ImageGenerationConfig
        if (!prompts.contentEquals(other.prompts)) return false
        if (negative_prompts != null) {
            if (other.negative_prompts == null) return false
            if (!negative_prompts.contentEquals(other.negative_prompts)) return false
        } else if (other.negative_prompts != null) return false
        if (height != other.height) return false
        if (width != other.width) return false
        if (sampler_config != other.sampler_config) return false
        if (scheduler_config != other.scheduler_config) return false
        if (strength != other.strength) return false

        return true
    }

    override fun hashCode(): Int {
        var result = prompts.contentHashCode()
        result = 31 * result + (negative_prompts?.contentHashCode() ?: 0)
        result = 31 * result + height!!
        result = 31 * result + width!!
        result = 31 * result + sampler_config.hashCode()
        result = 31 * result + scheduler_config.hashCode()
        result = 31 * result + strength.hashCode()
        return result
    }
}
