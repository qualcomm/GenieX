package com.geniex.sdk.bean

data class SchedulerConfig(
    /* Scheduler type: "ddim", etc. */
    val type: String? = "ddim",
    /* Training timesteps */
    val num_train_timesteps: Int? = 1000,
    /* An offset added to the inference steps */
    val steps_offset: Int? = 1,
    /* Beta schedule start */
    val beta_start: Float? = 0.00085f,
    /* Beta schedule end */
    val beta_end: Float? = 0.012f,
    /* Beta schedule: "scaled_linear" */
    val beta_schedule: String? = "scaled_linear",
    /* Prediction type: "epsilon", "v_prediction" */
    val prediction_type: String? = "epsilon",
    /* Timestep type: "discrete", "continuous" */
    val timestep_type: String? = "discrete",
    /* Timestep spacing: "linspace", "leading", "trailing" */
    val timestep_spacing: String? = "leading",
    /* Interpolation type: "linear", "exponential" */
    val interpolation_type: String? = "linear",
    /* Optional config file path */
    val config_path: String? = null
)
