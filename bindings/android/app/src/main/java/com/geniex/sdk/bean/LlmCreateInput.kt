package com.geniex.sdk.bean

data class LlmCreateInput(
    override val model_path: String,
    val tokenizer_path: String? = null,
    override val config: ModelConfig,
    /**
     * [RuntimeIdValue] to use for the model
     */
    override val runtime_id: String? = null,
    /**
     * Compute unit alias. `null` selects the runtime default ([ComputeUnitValue.HYBRID]
     * for `llama_cpp`, [ComputeUnitValue.NPU] for `qairt`). Use
     * [ComputeUnitValue.CPU] / [ComputeUnitValue.GPU] / [ComputeUnitValue.NPU] /
     * [ComputeUnitValue.HYBRID] to pin a specific compute unit.
     */
    override val compute_unit: String? = null,
) : CreateInputBase
