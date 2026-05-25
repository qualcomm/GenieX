package com.geniex.demo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.AdapterView
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.ProgressBar
import android.widget.SimpleAdapter
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.FragmentActivity
import androidx.recyclerview.widget.RecyclerView
import com.gyf.immersionbar.ktx.immersionBar
import com.hjq.toast.Toaster
import com.geniex.demo.bean.ModelData
import com.geniex.demo.bean.getSupportPluginIds
import com.geniex.demo.bean.isNpuModel
import com.geniex.demo.databinding.ActivityMainBinding
import com.geniex.demo.databinding.DialogSelectPluginIdBinding
import com.geniex.demo.listeners.CustomDialogInterface
import com.geniex.demo.utils.ExecShell
import com.geniex.demo.utils.ImgUtil
import com.geniex.demo.utils.WavRecorder
import com.geniex.demo.utils.inflate
import com.geniex.sdk.AsrWrapper
import com.geniex.sdk.CvWrapper
import com.geniex.sdk.EmbedderWrapper
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.GeniexSdk
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.RerankerWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.AsrCreateInput
import com.geniex.sdk.bean.AsrTranscribeInput
import com.geniex.sdk.bean.CVCapability
import com.geniex.sdk.bean.CVCreateInput
import com.geniex.sdk.bean.CVModelConfig
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.EmbedderCreateInput
import com.geniex.sdk.bean.EmbeddingConfig
import com.geniex.sdk.bean.HubSource
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.ModelPullInput
import com.geniex.sdk.bean.RerankConfig
import com.geniex.sdk.bean.RerankerCreateInput
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmContent
import com.geniex.sdk.bean.VlmCreateInput
import com.geniex.sdk.bean.DeviceIdValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream

class MainActivity : FragmentActivity() {

    private val binding: ActivityMainBinding by inflate()
    private var downloadJob: Job? = null
    private var downloadingModelData: ModelData? = null
    private lateinit var llDownloading: LinearLayout
    private lateinit var tvDownloadProgress: TextView
    private lateinit var pbDownloading: ProgressBar
    private lateinit var spModelList: Spinner
    private lateinit var btnDownload: Button
    private lateinit var btnLoadModel: Button
    private lateinit var btnUnloadModel: Button
    private lateinit var btnStop: Button
    private lateinit var etInput: EditText
    private lateinit var btnSend: Button
    private lateinit var btnClearHistory: Button
    private lateinit var btnAddImage: Button
    private lateinit var btnAudioRecord: Button

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ChatAdapter
    private lateinit var bottomPanel: LinearLayout
    private lateinit var btnAudioDone: Button
    private lateinit var btnAudioCancel: Button

    private lateinit var scrollImages: HorizontalScrollView
    private lateinit var topScrollContainer: LinearLayout
    private lateinit var llLoading: LinearLayout
    private lateinit var vTip: View

    private lateinit var llmWrapper: LlmWrapper
    private lateinit var vlmWrapper: VlmWrapper
    var embedderWrapper: EmbedderWrapper? = null
    private lateinit var rerankerWrapper: RerankerWrapper
    private lateinit var cvWrapper: CvWrapper
    private lateinit var asrWrapper: AsrWrapper
    private val modelScope = CoroutineScope(Dispatchers.IO)

    private val chatList = arrayListOf<ChatMessage>()
    private lateinit var llmSystemPrompt: ChatMessage
    private val vlmChatList = arrayListOf<VlmChatMessage>()
    private lateinit var vlmSystemPrompty: VlmChatMessage
    private lateinit var modelList: List<ModelData>
    private var selectModelId = ""

    // ADD: Track which model type is loaded
    private var isLoadLlmModel = false
    private var isLoadVlmModel = false
    private var isLoadEmbedderModel = false
    private var isLoadRerankerModel = false
    private var isLoadCVModel = false
    private var isLoadAsrModel = false

    private var enableThinking = false

    private var wavRecorder: WavRecorder? = null
    private var audioFile: File? = null

    private val savedImageFiles = mutableListOf<File>()
    private val messages = arrayListOf<Message>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        immersionBar {
            statusBarColorInt(Color.WHITE)
            statusBarDarkFont(true)
        }
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1002)
        initData()
        initView()
        setListeners()
    }

    private fun resetLoadState() {
        isLoadLlmModel = false
        isLoadVlmModel = false
        isLoadEmbedderModel = false
        isLoadRerankerModel = false
        isLoadCVModel = false
        isLoadAsrModel = false
    }

    private fun initView() {
        adapter = ChatAdapter(messages)
        binding.rvChat.adapter = adapter

        llDownloading = findViewById(R.id.ll_downloading)
        tvDownloadProgress = findViewById(R.id.tv_download_progress)
        pbDownloading = findViewById(R.id.pb_downloading)
        spModelList = findViewById(R.id.sp_model_list)
        spModelList.adapter = object : SimpleAdapter(this, modelList.map {
            val map = mutableMapOf<String, String>()
            map["displayName"] = it.displayName
            map
        }, R.layout.item_model, arrayOf("displayName"), intArrayOf(R.id.tv_model_id)) {

        }
        spModelList.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?, view: View?, position: Int, id: Long
            ) {
                selectModelId = modelList[position].id

                messages.clear()
                adapter.notifyDataSetChanged()
                binding.rvChat.scrollTo(0, 0)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectModelId = ""
            }
        }
        btnDownload = findViewById(R.id.btn_download)
        btnLoadModel = findViewById(R.id.btn_load_model)
        btnUnloadModel = findViewById(R.id.btn_unload_model)
        btnStop = findViewById(R.id.btn_stop)
        etInput = findViewById(R.id.et_input)
        btnAddImage = findViewById(R.id.btn_add_image)
        btnAudioRecord = findViewById(R.id.btn_voice)

        bottomPanel = findViewById(R.id.bottom_panel)
        btnAudioCancel = findViewById(R.id.btn_audio_cancel)
        btnAudioDone = findViewById(R.id.btn_audio_done)

        btnSend = findViewById(R.id.btn_send)
        btnClearHistory = findViewById(R.id.btn_clear_history)
        scrollImages = findViewById(R.id.scroll_images)
        topScrollContainer = findViewById(R.id.ll_images_container)
        llLoading = findViewById(R.id.ll_loading)
        vTip = findViewById<View>(R.id.v_tip)

        btnAudioCancel.setOnClickListener {
            stopRecord(true)
        }

        btnAudioDone.setOnClickListener {
            stopRecord(false)
        }

        findViewById<Button>(R.id.btn_test).setOnClickListener {
            Thread {
                val exeFile = File(filesDir, "geniex_test_llm")
                val chmodProcess = Runtime.getRuntime().exec("chmod 755 " + exeFile.absolutePath);
                chmodProcess.waitFor()
                Log.d("nfl", "exeFile exe? ${exeFile.canExecute()}")
                Log.d("nfl", "Exe Thread:${Thread.currentThread().name}")
                ExecShell().executeCommand(
                    arrayOf(
                        //                        exeFile.absolutePath,
//                        "--test-suite=\"npu\"", "--success "
                        "cat",
                        "/sys/devices/soc0/sku"
//                        "/data/local/tmp/test_cat.txt"
                    )
                ).forEach {
                    Log.d("nfl", "cmd:$it")
                }
            }.start()
        }

        findViewById<View>(R.id.v_tip).setOnClickListener {
            Toast.makeText(this, "please unload model first", Toast.LENGTH_SHORT).show()
        }
    }

    private fun parseModelList() {
        try {
            val baseJson = assets.open("model_list.json").bufferedReader().use { it.readText() }
            val json = Json { ignoreUnknownKeys = true }
            modelList = json.decodeFromString<List<ModelData>>(baseJson)
        } catch (e: Exception) {
            Log.e("nfl", "parseModelList: $e")
        }
    }

    /**
     * Step 0. Parse the model list and initialise the SDK. Model presence
     * is queried from the Rust model manager, not tracked client-side.
     */
    private fun initData() {
        parseModelList()
        //
        initGeniexSdk()
        //
        val sysPrompt = """\
You are Nays Campaign Manager, an AI assistant responsible for managing customer campaigns and investigating campaign-related issues.

When a customer inquiry comes in, you need to:
1. Analyze the customer's request to understand their campaign needs
2. Check if it's related to campaign limits or issues
3. Use the campaign_investigation function when needed to check campaign status
4. Provide appropriate responses based on the investigation results

Your responsibilities include:
- Investigating campaign performance and limits
- Determining if customers have reached their campaign limits
- Providing helpful messages when limits are reached
- Directing customers to support when limits haven't been reached
- Ensuring smooth campaign operations for all customers

When you receive a query about campaigns, you should:
1. First understand what the customer is asking about
2. If it's campaign-related, use the campaign_investigation tool to check the status
3. Based on the tool's response, provide appropriate guidance

Always be professional, helpful, and focused on resolving campaign-related issues efficiently.

Note: You must use the campaign_investigation function whenever a customer asks about campaign limits, issues, or status.
"""
        // It works better with Chinese prompt words.
        val sysPrompt2 = "Must reply in markdown format"
//        addSystemPrompt(sysPrompt2)
    }

    /**
     * Step 1. initGeniexSdk environment
     */
    private fun initGeniexSdk() {
        // Initialize GeniexSdk with context
        GeniexSdk.getInstance().init(this, object : GeniexSdk.InitCallback {
            override fun onSuccess() {
            }

            override fun onFailure(reason: String) {
                Log.e(TAG, "GeniexSdk init failed: $reason")
            }
        })
    }

    /**
     * Step 2. add system prompt, such as : output markdown style, contains emoji etc.(Options)
     */
    private fun addSystemPrompt(sysPrompt: String) {
        llmSystemPrompt = ChatMessage("system", sysPrompt)
        chatList.add(llmSystemPrompt)
        vlmSystemPrompty =
            VlmChatMessage(
                "system",
                listOf(VlmContent("text", sysPrompt))
            )
        vlmChatList.add(vlmSystemPrompty)
    }


    private fun onLoadModelSuccess(tip: String) {
        runOnUiThread {
            Toast.makeText(
                this@MainActivity, tip, Toast.LENGTH_SHORT
            ).show()
            // change UI
            btnAddImage.visibility = View.INVISIBLE
            btnAudioRecord.visibility = View.INVISIBLE
            if (isLoadVlmModel) {
                btnAddImage.visibility = View.VISIBLE
                btnAudioRecord.visibility = View.VISIBLE
            }
            if (isLoadCVModel) {
                btnAddImage.visibility = View.VISIBLE
            }
            if (isLoadAsrModel) {
                btnAudioRecord.visibility = View.VISIBLE
            }
            //
            btnUnloadModel.visibility = View.VISIBLE
            llLoading.visibility = View.INVISIBLE
            //
            if (isLoadEmbedderModel || isLoadRerankerModel || isLoadAsrModel || isLoadCVModel) {
                btnStop.visibility = View.GONE
            } else {
                btnStop.visibility = View.VISIBLE
            }
        }
    }

    private fun onLoadModelFailed(tip: String) {
        runOnUiThread {
            vTip.visibility = View.GONE
            Toast.makeText(this@MainActivity, tip, Toast.LENGTH_SHORT).show()
            // change UI
            btnAddImage.visibility = View.INVISIBLE
            btnAudioRecord.visibility = View.INVISIBLE
            btnUnloadModel.visibility = View.GONE
            llLoading.visibility = View.INVISIBLE
        }
    }

    private fun hasLoadedModel(): Boolean {
        return isLoadLlmModel || isLoadVlmModel || isLoadEmbedderModel ||
                isLoadRerankerModel || isLoadCVModel || isLoadAsrModel
    }

    /**
     * Checks the Rust model manager's cache for [modelData]. The
     * manager filters out `.inflight/` entries, so a cancelled download
     * returns `false` here until it completes.
     */
    private suspend fun isModelDownloaded(modelData: ModelData): Boolean {
        return ModelManagerWrapper.list().contains(modelData.modelName)
    }

    /**
     * Draw bounding boxes on the image for object detection results
     */
    private fun drawBoundingBoxes(originalBitmap: Bitmap, results: List<com.geniex.sdk.bean.CVResult>): Bitmap {
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // Prepare paint for bounding boxes
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f
            isAntiAlias = true
        }

        // Prepare paint for text background
        val textBgPaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Prepare paint for text
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 20f
            isAntiAlias = true
            isFakeBoldText = true
        }

        // Color palette for different classes
        val colors = listOf(
            Color.rgb(255, 0, 0),     // Red
            Color.rgb(0, 255, 0),     // Green
            Color.rgb(0, 0, 255),     // Blue
            Color.rgb(255, 255, 0),   // Yellow
            Color.rgb(255, 0, 255),   // Magenta
            Color.rgb(0, 255, 255),   // Cyan
            Color.rgb(255, 128, 0),   // Orange
            Color.rgb(128, 0, 255)    // Purple
        )

        results.forEachIndexed { index, result ->
            result.bbox?.let { bbox ->
                // bbox already has pixel values (not normalized 0-1)
                val x1 = bbox.x
                val y1 = bbox.y
                val x2 = bbox.x + bbox.width
                val y2 = bbox.y + bbox.height

                // Select color based on index
                val color = colors[index % colors.size]
                boxPaint.color = color
                textBgPaint.color = color

                // Draw bounding box
                val rect = RectF(x1, y1, x2, y2)
                canvas.drawRect(rect, boxPaint)

                // Prepare label text
                val label = result.text ?: "object"
                val confidence = result.confidence ?: 0.0
                val labelText = "$label ${String.format("%.2f", confidence)}"

                // Measure text
                val textBounds = android.graphics.Rect()
                textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)

                // Draw text background
                val textBgRect = RectF(
                    x1,
                    y1 - textBounds.height() - 4f,
                    x1 + textBounds.width() + 8f,
                    y1
                )
                canvas.drawRect(textBgRect, textBgPaint)

                // Draw text
                canvas.drawText(labelText, x1 + 4f, y1 - 4f, textPaint)
            }
        }

        return mutableBitmap
    }

    private fun loadModel(
        selectModelData: ModelData,
        modelDataPluginId: String,
        nGpuLayers: Int,
        deviceId: String? = null
    ) {
        modelScope.launch {
            resetLoadState()
            val paths = ModelManagerWrapper.getPaths(selectModelData.modelName)
            if (paths == null) {
                onLoadModelFailed("model paths unavailable — pull it first")
                return@launch
            }
            // Manifest-written plugin_id wins when present; fall back to
            // the user's UI selection for GGUF models that skip the manifest.
            val pluginId = paths.plugin_id.ifEmpty { modelDataPluginId }
            val resolvedDeviceId = deviceId ?: paths.device_id
            when (selectModelData.type) {
                "chat", "llm" -> {
                    // QAIRT rejects non-zero n_ctx / n_gpu_layers (both fixed at compile
                    // time in the AI Hub bundle) — and the Kotlin ModelConfig defaults
                    // are non-zero, so zero them explicitly for the qairt path.
                    val isQairt = pluginId == "qairt"
                    val conf = if (isQairt) {
                        ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = enableThinking)
                    } else {
                        ModelConfig(
                            nCtx = 1024,
                            nGpuLayers = nGpuLayers,
                            enable_thinking = enableThinking,
                        )
                    }
                    LlmWrapper.builder().llmCreateInput(
                        LlmCreateInput(
                            model_name = paths.model_name,
                            model_path = paths.model_path,
                            tokenizer_path = paths.tokenizer_path,
                            config = conf,
                            plugin_id = pluginId,
                            device_id = resolvedDeviceId ?: DeviceIdValue.NPU.value,
                        )
                    ).build().onSuccess { wrapper ->
                        isLoadLlmModel = true
                        llmWrapper = wrapper
                        onLoadModelSuccess("llm model loaded")
                    }.onFailure { error ->
                        onLoadModelFailed(error.message.toString())
                    }
                }

                "embedder" -> {
                    EmbedderWrapper.builder()
                        .embedderCreateInput(
                            EmbedderCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                tokenizer_path = paths.tokenizer_path,
                                config = ModelConfig(nGpuLayers = nGpuLayers),
                                plugin_id = pluginId,
                                device_id = resolvedDeviceId ?: DeviceIdValue.CPU.value,
                            )
                        )
                        .build().onSuccess { wrapper ->
                            isLoadEmbedderModel = true
                            embedderWrapper = wrapper
                            onLoadModelSuccess("embedder model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                "reranker" -> {
                    RerankerWrapper.builder()
                        .rerankerCreateInput(
                            RerankerCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                tokenizer_path = paths.tokenizer_path,
                                config = ModelConfig(nGpuLayers = nGpuLayers),
                                plugin_id = pluginId,
                                device_id = resolvedDeviceId ?: DeviceIdValue.CPU.value,
                            )
                        )
                        .build().onSuccess { wrapper ->
                            isLoadRerankerModel = true
                            rerankerWrapper = wrapper
                            onLoadModelSuccess("reranker model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                "cv" -> {
                    CvWrapper.builder()
                        .createInput(
                            CVCreateInput(
                                model_name = paths.model_name,
                                config = CVModelConfig(
                                    capabilities = CVCapability.OCR,
                                    det_model_path = paths.model_dir,
                                    rec_model_path = paths.model_path,
                                    char_dict_path = paths.model_dir,
                                ),
                                plugin_id = pluginId,
                            )
                        )
                        .build().onSuccess {
                            isLoadCVModel = true
                            cvWrapper = it
                            onLoadModelSuccess("cv model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                "asr" -> {
                    AsrWrapper.builder()
                        .asrCreateInput(
                            AsrCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                config = ModelConfig(nGpuLayers = nGpuLayers),
                                plugin_id = pluginId,
                            )
                        )
                        .build().onSuccess { wrapper ->
                            isLoadAsrModel = true
                            asrWrapper = wrapper
                            onLoadModelSuccess("ASR model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                "multimodal", "vlm" -> {
                    val isNpuVlm = pluginId == "qairt"
                    val config = if (isNpuVlm) {
                        // QAIRT rejects non-zero n_ctx / n_gpu_layers for VLM too.
                        ModelConfig(nCtx = 0, nGpuLayers = 0, nThreads = 8, enable_thinking = enableThinking)
                    } else {
                        ModelConfig(
                            nCtx = 1024,
                            nThreads = 4,
                            nBatch = 1,
                            nUBatch = 1,
                            nGpuLayers = nGpuLayers,
                            enable_thinking = enableThinking,
                        )
                    }
                    VlmWrapper.builder()
                        .vlmCreateInput(
                            VlmCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                mmproj_path = paths.mmproj_path,
                                config = config,
                                plugin_id = pluginId,
                                device_id = resolvedDeviceId ?: "HTP0",
                            )
                        )
                        .build().onSuccess {
                            isLoadVlmModel = true
                            vlmWrapper = it
                            onLoadModelSuccess("vlm model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                else -> {
                    onLoadModelFailed("model type error")
                }
            }
        }
    }

    private fun downloadModel(selectModelData: ModelData) {
        if (hasLoadedModel()) {
            Toast.makeText(this@MainActivity, "unload the current model first", Toast.LENGTH_SHORT).show()
            return
        }
        if (downloadJob?.isActive == true) {
            Toaster.show("${downloadingModelData?.displayName ?: "a model"} is already downloading")
            return
        }

        downloadingModelData = selectModelData
        llDownloading.visibility = View.VISIBLE
        tvDownloadProgress.text = "0%"

        val hub = runCatching { HubSource.valueOf(selectModelData.hub ?: "AUTO") }
            .getOrDefault(HubSource.AUTO)
        // AI Hub pulls route through chipset-matched assets. The Rust side
        // can auto-detect the host only on Windows-on-Snapdragon, so on
        // Android we must pass an explicit chipset for anything that ends
        // up on the AI Hub path — whether hub is AIHUB or AUTO + qualcomm/*.
        val willUseAiHub = hub == HubSource.AIHUB ||
            (hub == HubSource.AUTO && selectModelData.modelName.startsWith("qualcomm/"))
        if (willUseAiHub && selectModelData.chipset.isNullOrBlank()) {
            llDownloading.visibility = View.GONE
            Toaster.show("AI Hub models require a chipset. Update model_list.json.")
            return
        }
        val input = ModelPullInput(
            model_name = selectModelData.modelName,
            quant = selectModelData.quant,
            hub = hub,
            chipset = selectModelData.chipset,
            display_name = selectModelData.aiHubDisplayName,
        )

        downloadJob = modelScope.launch {
            // Short-circuit if already cached — the manager filters .inflight/
            // models out of list(), so this only matches a complete pull.
            if (isModelDownloaded(selectModelData)) {
                runOnUiThread {
                    llDownloading.visibility = View.GONE
                    Toast.makeText(this@MainActivity, "model already downloaded", Toast.LENGTH_SHORT).show()
                }
                return@launch
            }

            ModelManagerWrapper.pullFlow(input).collect { event ->
                when (event) {
                    is ModelManagerWrapper.PullEvent.Progress -> {
                        val total = event.files.sumOf { if (it.total_bytes > 0) it.total_bytes else 0L }
                        val done = event.files.sumOf { it.downloaded_bytes }
                        val percent = if (total > 0) ((done * 100) / total).toInt() else 0
                        runOnUiThread { tvDownloadProgress.text = "$percent%" }
                    }
                    is ModelManagerWrapper.PullEvent.Completed -> {
                        runOnUiThread {
                            llDownloading.visibility = View.GONE
                            Toaster.show("${selectModelData.displayName} downloaded")
                        }
                    }
                    is ModelManagerWrapper.PullEvent.Error -> {
                        Log.e(TAG, "pull failed rc=${event.code}: ${event.message}")
                        runOnUiThread {
                            llDownloading.visibility = View.GONE
                            Toaster.showLong("Download failed: ${event.message}")
                        }
                    }
                }
            }
        }
    }

    private fun setListeners() {

        btnAddImage.setOnClickListener {
            openGallery()
        }

        btnAudioRecord.setOnClickListener {
            startRecord()
        }

        btnClearHistory.setOnClickListener {
            clearHistory()
        }
        /**
         * Step 3. download model. Cancelling the coroutine triggers the
         * flow's awaitClose which flips the Rust progress callback to
         * return false — partial files stay on disk for a resumed pull.
         * Use the Retry button to kick off a fresh pull that resumes.
         */
        binding.btnCancelDownload.setOnClickListener {
            downloadJob?.cancel()
            downloadJob = null
            tvDownloadProgress.text = "0%"
            binding.btnDismissDownload.performClick()
        }
        binding.btnRetryDownload.setOnClickListener {
            downloadJob?.cancel()
            downloadJob = null
            downloadingModelData?.let { downloadModel(it) }
        }
        binding.btnDismissDownload.setOnClickListener {
            binding.llDownloading.visibility = View.GONE
        }
        btnDownload.setOnClickListener {
            if (downloadJob?.isActive == true) {
                if (downloadingModelData?.id == selectModelId) {
                    binding.llDownloading.visibility = View.VISIBLE
                } else {
                    Toaster.show("${downloadingModelData?.displayName} is currently downloading.")
                }
                return@setOnClickListener
            }
            val selectModelData = modelList.first { it.id == selectModelId }
            downloadModel(selectModelData)
        }
        /**
         * Step 4. load model
         */
        btnLoadModel.setOnClickListener {
            val selectModelData = modelList.first { it.id == selectModelId }
            if (selectModelData == null) {
                Toast.makeText(this@MainActivity, "model not selected", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            Log.d(TAG, "current select model data:$selectModelData")
            if (hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "please unload first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Availability is checked against the manager's cache — a pull
            // that was cancelled mid-flight is not listed until it completes.
            modelScope.launch {
                if (!isModelDownloaded(selectModelData)) {
                    runOnUiThread {
                        Toaster.showLong("Model not downloaded — tap Download first.")
                    }
                    return@launch
                }
                runOnUiThread { startLoadModel(selectModelData) }
            }
        }

        /**
         * Step 5. send message
         */
        btnSend.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "please load model first", Toast.LENGTH_SHORT)
                    .show()
                return@setOnClickListener
            }

            if (savedImageFiles.isNotEmpty()) {
                messages.add(Message("", MessageType.IMAGES, savedImageFiles.map { it }))
                reloadRecycleView()
            }

            val inputString = etInput.text.trim().toString()
            etInput.setText("")
            etInput.clearFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(etInput.windowToken, 0)

            if (inputString.isNotEmpty()) {
                messages.add(Message(inputString, MessageType.USER))
                reloadRecycleView()
            }

            val supportFunctionCall = false
            var tools: String? = null
            var grammarString: String? = null
            if (supportFunctionCall) {
                // if this model support 'function call'
                tools =
                    "[{\"type\":\"function\",\"function\":{\"name\": \"campaign_investigation\",\"description\": \"Check campaign limits and determine appropriate action. If customer has reached limit, return a message (hardcoded or generated by model). If limit not reached, contact support.\",\"parameters\": {\"type\": \"object\", \"properties\":{\"campaign_name\":{\"type\": \"string\",\"description\": \"The name of the campaign to investigate\"}}, \"required\":[\"campaign_name\"]}}}]"
                grammarString = """
root ::= "<tool_call>" space object "</tool_call>" space
object ::= "{" space campaign-name-kv "}" space
campaign-name-kv ::= "\"campaign_name\"" space ":" space string
string ::= "\"" char* "\"" space
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
space ::= | " " | "\n" | "\r" | "\t"
"""
            }

            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "model not loaded", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            modelScope.launch {
                val selectModelData = modelList.first { it.id == selectModelId }
                val isNpu = ModelManagerWrapper.getPaths(selectModelData.modelName)?.plugin_id == "qairt"
                Log.d(TAG, "isNpu: $isNpu")

                val sb = StringBuilder()
                if (isLoadCVModel) {
                    // FIXME: Temporarily select the last image
                    if (savedImageFiles.isEmpty()) {
                        runOnUiThread {
                            Toast.makeText(
                                this@MainActivity,
                                "Please select one picture.",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                        return@launch
                    }
                    val imagePath = savedImageFiles.last().absolutePath
                    runOnUiThread {
                        messages.add(Message("", MessageType.IMAGES, savedImageFiles))
                        reloadRecycleView()
                        clearImages()
                    }
                    cvWrapper.infer(imagePath).onSuccess { results ->
                        Log.d("nfl", "infer result:$results")
                        runOnUiThread {
                            val outputImageFiles = results.flatMap { r ->
                                r.image_paths?.map { File(it) }?.filter { it.exists() } ?: emptyList()
                            }
                            val isObjectDetection = results.firstOrNull()?.bbox != null
                            Log.d(
                                "nfl",
                                "outputImageFiles: ${outputImageFiles.size}, isObjectDetection: $isObjectDetection, results count: ${results.size}"
                            )

                            when {
                                outputImageFiles.isNotEmpty() -> {
                                    messages.add(Message("", MessageType.ASSISTANT_IMAGES, outputImageFiles))
                                }

                                isObjectDetection -> {
                                    try {
                                        val originalBitmap = BitmapFactory.decodeFile(imagePath)
                                        Log.d("nfl", "Loaded bitmap: $originalBitmap")
                                        if (originalBitmap == null) {
                                            messages.add(
                                                Message(
                                                    "Failed to load image: $imagePath",
                                                    MessageType.PROFILE
                                                )
                                            )
                                            reloadRecycleView()
                                            return@runOnUiThread
                                        }
                                        val resultBitmap = drawBoundingBoxes(originalBitmap, results)
                                        Log.d("nfl", "Drew bounding boxes: $resultBitmap")
                                        val resultFile =
                                            File(filesDir, "detection_result_${System.currentTimeMillis()}.jpg")
                                        resultFile.outputStream().use { out ->
                                            resultBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                                        }
                                        Log.d("nfl", "Saved detection result to: ${resultFile.absolutePath}")
                                        messages.add(Message("", MessageType.ASSISTANT_IMAGES, listOf(resultFile)))
                                        val summary = results.mapIndexed { idx, result ->
                                            val label = result.text ?: "object"
                                            val conf = String.format("%.2f", result.confidence ?: 0.0)
                                            "${idx + 1}. $label ($conf)"
                                        }.joinToString("\n")
                                        messages.add(
                                            Message(
                                                "Detected ${results.size} objects:\n$summary",
                                                MessageType.ASSISTANT
                                            )
                                        )
                                    } catch (e: Exception) {
                                        Log.e("nfl", "Error drawing bounding boxes", e)
                                        messages.add(
                                            Message(
                                                "Error processing detection: ${e.message}",
                                                MessageType.PROFILE
                                            )
                                        )
                                    }
                                }

                                else -> {
                                    Log.d("nfl", "Processing as OCR, results: $results")
                                    val content = results.map { result ->
                                        "[${result.confidence}] ${result.text}"
                                    }.joinToString(separator = "\n")
                                    messages.add(Message(content, MessageType.ASSISTANT))
                                }
                            }
                            reloadRecycleView()
                        }
                    }.onFailure { error ->
                        runOnUiThread {
                            messages.add(Message(error.toString(), MessageType.PROFILE))
                            reloadRecycleView()
                        }
                        Log.d("nfl", "infer result error:$error")
                    }
                } else if (isLoadAsrModel) {
                    if (audioFile == null) {
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "no audio file", Toast.LENGTH_SHORT)
                                .show()
                        }
                    } else {
//                        val audioFilePath = audioFile!!.absolutePath
                        val audioFilePath = "/sdcard/Download/assets/OSR_us_000_0010_16k.wav"
                        asrWrapper.transcribe(
                            AsrTranscribeInput(
                                audioFilePath,  // Use hardcoded path instead of inputString
                                "en",  // Language code
                                null   // Optional timestamps
                            )
                        ).onSuccess { transcription ->
                            runOnUiThread {
                                messages.add(
                                    Message(
                                        transcription.result.transcript ?: "",
                                        MessageType.ASSISTANT
                                    )
                                )
                                reloadRecycleView()
                            }
                        }.onFailure { error ->
                            runOnUiThread {
                                messages.add(
                                    Message(
                                        "Error: ${error.message}",
                                        MessageType.PROFILE
                                    )
                                )
                                reloadRecycleView()
                            }
                        }
                    }
                } else if (isLoadEmbedderModel) {
                    // ADD: Handle embedder inference
                    // Input format: single text or multiple texts separated by "|"
                    val texts = inputString.split("|").map { it.trim() }.toTypedArray()
                    embedderWrapper!!.embed(texts, EmbeddingConfig()).onSuccess { embedResult ->
                        runOnUiThread {
                            val result = StringBuilder()
                            val flatEmbeddings = embedResult.embeddings
                            val embeddingDim = flatEmbeddings.size / texts.size

                            texts.forEachIndexed { idx, text ->
                                val start = idx * embeddingDim
                                val end = start + embeddingDim
                                val embedding = flatEmbeddings.sliceArray(start until end)

                                // Calculate mean and variance
                                val mean = embedding.average()
                                val variance = embedding.map { (it - mean) * (it - mean) }.average()

                                result.append("Text ${idx + 1}: \"$text\"\n")
                                result.append("Embedding dimension: $embeddingDim\n")
                                result.append("Mean: ${"%.4f".format(mean)}\n")
                                result.append("Variance: ${"%.4f".format(variance)}\n")
                                result.append("First 5 values: [")
                                result.append(
                                    embedding.take(5).joinToString(", ") { "%.4f".format(it) })
                                result.append("...]\n\n")
                            }

                            messages.add(Message(result.toString(), MessageType.ASSISTANT))
                            reloadRecycleView()
                        }
                    }.onFailure { error ->
                        runOnUiThread {
                            messages.add(Message("Error: ${error.message}", MessageType.PROFILE))
                            reloadRecycleView()
                        }
                    }
                } else if (isLoadRerankerModel) {
                    // Reranker input format: "query\ndoc1\ndoc2\ndoc3..."
                    // First line is query, remaining lines are documents
                    val query = inputString.split("\n")[0]  // Get first line as query
                    val documents =
                        inputString.split("\n").drop(1).toTypedArray()  // Get rest as docs
                    rerankerWrapper.rerank(query, documents, RerankConfig())
                        .onSuccess { rerankerResult ->
                            runOnUiThread {
                                val result = StringBuilder()
                                result.append("Rerank Results:\n")
                                // Sort by score descending to show best matches first
                                rerankerResult.scores?.withIndex()?.sortedByDescending { it.value }
                                    ?.forEach { (idx, score) ->
                                        result.append("${idx + 1}. Score: ${"%.4f".format(score)}\n")
                                        result.append("   ${documents[idx]}\n\n")
                                    }
                                messages.add(Message(result.toString(), MessageType.ASSISTANT))
                                reloadRecycleView()
                            }
                        }.onFailure { error ->
                            runOnUiThread {
                                "Error: ${error.message}".also {
                                    messages.add(Message(it, MessageType.PROFILE))
                                    reloadRecycleView()
                                }
                            }
                        }
                } else if (isLoadVlmModel) {
                    val contents = savedImageFiles.map {
                        VlmContent("image", it.absolutePath)
                    }.toMutableList()
                    audioFile?.let {
                        contents.add(VlmContent("audio", it.absolutePath))
                    }
                    contents.add(VlmContent("text", inputString))
                    audioFile = null
                    clearImages()
                    val sendMsg = VlmChatMessage(role = "user", contents = contents)
                    // VlmContentTransfer(
                    //     this@MainActivity, VlmContent(
                    //         "image", inputString
                    //     )
                    // ).forUrl()

                    // vlmChatList.clear()
                    vlmChatList.add(sendMsg)

                    Log.d(TAG, "before apply chat template:$vlmChatList")
                    vlmWrapper.applyChatTemplate(vlmChatList.toTypedArray(), tools, enableThinking)
                        .onSuccess { result ->
                            Log.d(TAG, "vlm chat template:${result.formattedText}")
                            val baseConfig =
                                GenerationConfigSample().toGenerationConfig(grammarString)
                            val configWithMedia = vlmWrapper.injectMediaPathsToConfig(
                                vlmChatList.toTypedArray(),
                                baseConfig
                            )

                            Log.d(TAG, "Config has ${configWithMedia.imageCount} images")

                            vlmWrapper.generateStreamFlow(
                                result.formattedText,
                                configWithMedia
                            ).collect { handleResult(sb, it) }
                        }.onFailure {
                            runOnUiThread {
                                Toast.makeText(
                                    this@MainActivity, it.message, Toast.LENGTH_SHORT
                                ).show()
                            }
                        }
                } else {
                    chatList.add(ChatMessage(role = "user", inputString))
                    // Apply chat template and generate
                    llmWrapper.applyChatTemplate(
                        chatList.toTypedArray(),
                        tools,
                        enableThinking
                    ).onSuccess { templateOutput ->
                        Log.d(TAG, "chat template:${templateOutput.formattedText}")
                        llmWrapper.generateStreamFlow(
                            templateOutput.formattedText,
                            GenerationConfigSample().toGenerationConfig(grammarString)
                        ).collect { streamResult ->
                            handleResult(sb, streamResult)
                        }
                    }.onFailure { error ->
                        runOnUiThread {
                            Toast.makeText(
                                this@MainActivity, error.message, Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                }

                clearImages()
            }

        }

        /**
         * Step 6. others
         */
        btnUnloadModel.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "model not loaded", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            // Unload model and cleanup
            val handleUnloadResult = fun(result: Int) {
                resetLoadState()
                runOnUiThread {
                    vTip.visibility = View.GONE
                    btnUnloadModel.visibility = View.GONE
                    btnStop.visibility = View.GONE
                    btnAddImage.visibility = View.INVISIBLE
                    btnAudioRecord.visibility = View.INVISIBLE
                    Toast.makeText(
                        this@MainActivity, if (result == 0) {
                            "unload success"
                        } else {
                            "unload failed and error code: $result"
                        }, Toast.LENGTH_SHORT
                    ).show()
                }
            }
            modelScope.launch {
                if (isLoadVlmModel) {
                    vlmWrapper.stopStream()
                    vlmWrapper.destroy()
                    vlmChatList.clear()
                    // TODO:
                    handleUnloadResult(0)
                } else if (isLoadEmbedderModel) {
                    // ADD: Unload embedder
                    embedderWrapper!!.destroy()
                    handleUnloadResult(0)
                } else if (isLoadRerankerModel) {
                    // ADD: Unload reranker
                    handleUnloadResult(rerankerWrapper.destroy())
                } else if (isLoadCVModel) {
                    // ADD: Unload CV model
                    cvWrapper.destroy()
                    // TODO:
                    handleUnloadResult(0)
                } else if (isLoadAsrModel) {
                    // ADD: Unload ASR model
                    asrWrapper.destroy()
                    // TODO:
                    handleUnloadResult(0)
                } else if (isLoadLlmModel) {
                    llmWrapper.stopStream()
                    llmWrapper.destroy()
                    chatList.clear()
                    // TODO:
                    handleUnloadResult(0)
                } else {
                    handleUnloadResult(0)
                }
            }
        }
        btnStop.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast.makeText(
                    this@MainActivity,
                    "model not loaded",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }
            // MODIFY: Stop button only works for LLM/VLM (not embedder/reranker)
            if (isLoadEmbedderModel || isLoadRerankerModel || isLoadAsrModel || isLoadCVModel) {
                Toast.makeText(
                    this@MainActivity,
                    "Stop not applicable for embedder/reranker/asr/cv",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }
            // Stop streaming
            modelScope.launch {
                if (isLoadVlmModel) {
                    vlmWrapper.stopStream()
                } else if (isLoadLlmModel) {
                    llmWrapper.stopStream()
                }
            }
        }
    }

    private fun startLoadModel(selectModelData: ModelData) {
        vTip.visibility = View.VISIBLE
        llLoading.visibility = View.VISIBLE

        val supportPluginIds = selectModelData.getSupportPluginIds()
        Log.d(TAG, "support plugin_id:$supportPluginIds")
        var modelDataPluginId = "llama_cpp"
        var nGpuLayers = 0
        if (supportPluginIds.size > 1) {
            val dialogBinding = DialogSelectPluginIdBinding.inflate(layoutInflater)
            val isGgufLlmModel = !selectModelData.isNpuModel() &&
                    (selectModelData.type == "chat" || selectModelData.type == "llm")
            supportPluginIds.forEach {
                when (it) {
                    "cpu" -> {
                        dialogBinding.rbCpu.visibility = View.VISIBLE
                        dialogBinding.rbCpu.isChecked = true
                    }
                    "gpu" -> {
                        dialogBinding.rbGpu.visibility = View.VISIBLE
                    }
                    "npu" -> {
                        dialogBinding.rbNpu.visibility = View.VISIBLE
                        dialogBinding.rbNpu.isChecked = true
                    }
                }
            }
            if (isGgufLlmModel) {
                dialogBinding.rbNpu.visibility = View.VISIBLE
            }
            dialogBinding.rgSelectPluginId.setOnCheckedChangeListener { _, checkedId ->
                dialogBinding.llGpuLayers.visibility =
                    if (checkedId == R.id.rb_gpu) View.VISIBLE else View.GONE
            }

            val dialogOnClickListener = object : CustomDialogInterface.OnClickListener() {
                override fun onClick(dialog: DialogInterface?, which: Int) {
                    nGpuLayers = 0
                    var ggufLlmDeviceId: String? = null
                    val checkedId = dialogBinding.rgSelectPluginId.checkedRadioButtonId
                    if (checkedId == R.id.rb_gpu) {
                        if (dialogBinding.llGpuLayers.visibility == View.VISIBLE) {
                            nGpuLayers = dialogBinding.etGpuLayers.text.toString().toInt()
                            if (nGpuLayers == 0) {
                                Toast.makeText(
                                    this@MainActivity,
                                    "nGpuLayers min value is 1",
                                    Toast.LENGTH_SHORT
                                ).show()
                                return
                            }
                        }
                        ggufLlmDeviceId = DeviceIdValue.GPU.value
                    } else if (checkedId == R.id.rb_npu && isGgufLlmModel) {
                        nGpuLayers = 999
                        ggufLlmDeviceId = DeviceIdValue.NPU.value
                    }
                    when (which) {
                        DialogInterface.BUTTON_POSITIVE -> {
                            dialog?.dismiss()
                            loadModel(selectModelData, modelDataPluginId, nGpuLayers, ggufLlmDeviceId)
                        }
                        DialogInterface.BUTTON_NEGATIVE -> {
                            llLoading.visibility = View.INVISIBLE
                            vTip.visibility = View.GONE
                        }
                    }
                }
            }
            val alertDialog = AlertDialog.Builder(this).setView(dialogBinding.root)
                .setNegativeButton("Cancel", dialogOnClickListener)
                .setPositiveButton("OK", dialogOnClickListener)
                .setCancelable(false)
                .create()
            alertDialog.show()
            dialogOnClickListener.resetPositiveButton(alertDialog)
        } else {
            if ("npu" == supportPluginIds[0]) {
                modelDataPluginId = "npu"
            }
            loadModel(selectModelData, modelDataPluginId, nGpuLayers)
        }
    }

    fun handleResult(sb: StringBuilder, streamResult: LlmStreamResult) {
        when (streamResult) {
            is LlmStreamResult.Token -> {
                runOnUiThread {
                    sb.append(streamResult.text)
                    Message(sb.toString(), MessageType.ASSISTANT).let { lastMsg ->
                        val size = messages.size
                        messages[size - 1].let { msg ->
                            if (msg.type != MessageType.ASSISTANT) {
                                messages.add(lastMsg)
                            } else {
                                messages[size - 1] = lastMsg
                            }
                        }
                    }
                    adapter.notifyDataSetChanged()
                }
                Log.d(TAG, "Token: ${streamResult.text}")
            }

            is LlmStreamResult.Completed -> {
                if (isLoadVlmModel) {
                    vlmChatList.add(
                        VlmChatMessage(
                            "assistant",
                            listOf(VlmContent("text", sb.toString()))
                        )
                    )
                } else {
                    chatList.add(ChatMessage("assistant", sb.toString()))
                }

                runOnUiThread {
                    var content = sb.toString()
                    val size = messages.size
                    messages[size - 1] = Message(content, MessageType.ASSISTANT)

                    val ttft = String.format(null, "%.2f", streamResult.profile.ttftMs)
                    val promptTokens = streamResult.profile.promptTokens
                    val prefillSpeed =
                        String.format(null, "%.2f", streamResult.profile.prefillSpeed)

                    val generatedTokens = streamResult.profile.generatedTokens
                    val decodingSpeed =
                        String.format(null, "%.2f", streamResult.profile.decodingSpeed)

                    val profileData =
                        "TTFT: $ttft ms; Prompt Tokens: $promptTokens; \nPrefilling Speed: $prefillSpeed tok/s\nGenerated Tokens: $generatedTokens; Decoding Speed: $decodingSpeed tok/s"
                    messages.add(
                        Message(
                            profileData,
                            MessageType.PROFILE
                        )
                    )
                    reloadRecycleView()
                }
                Log.d(TAG, "Completed: ${streamResult.profile}")
            }

            is LlmStreamResult.Error -> {
                runOnUiThread {
                    val content =
                        "your conversation is out of model’s context length, please start a new conversation or click clear button"
                    messages.add(Message(content, MessageType.PROFILE))
                    reloadRecycleView()
                }
                Log.d(TAG, "Error: $streamResult")
            }
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, null)
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*")
        startActivityForResult(intent, 1)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 0) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery()
            } else {
                Toast.makeText(this, "Not allow", Toast.LENGTH_SHORT).show()
            }
        } else if (requestCode == 2001) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, "Camera not allow", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        var bitmap: Bitmap? = null
        if (requestCode == 1) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                val inputStream = contentResolver.openInputStream(data.data!!)
                bitmap = BitmapFactory.decodeStream(inputStream)
            }
        } else if (requestCode == 1001 && resultCode == Activity.RESULT_OK) {
            photoFile?.let {
                bitmap = BitmapFactory.decodeFile(it.absolutePath)
            }
        }

        bitmap?.let {
            try {
                val file = File(filesDir, "chat_${System.currentTimeMillis()}.jpg")
                val success = saveBitmapToFile(it, file)
                if (success) {
                    Log.d(TAG, "Save success：${file.absolutePath}")
                    savedImageFiles.add(file)
                    refreshTopScrollContainer()
                } else {
                    Toast.makeText(this, "Save Image failed", Toast.LENGTH_SHORT).show()
                }
            } catch (e: FileNotFoundException) {
                e.printStackTrace()
            }
        }
    }

    private fun saveBitmapToFile(bitmap: Bitmap, file: File): Boolean {
        return try {
            val tempDir = File(this.filesDir, "tmp").apply { if (!exists()) mkdirs() }

            val tempFile = File(
                tempDir,
                "tmp_${System.currentTimeMillis()}.jpg"
            )
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }

            val outFile = File(
                tempDir,
                "out_${System.currentTimeMillis()}.jpg"
            )
            ImgUtil.squareCrop(
                ImgUtil.downscaleAndSave(
                    imageFile = tempFile,
                    outFile = outFile,
                    maxSize = 448,
                    format = Bitmap.CompressFormat.JPEG,
                    quality = 90
                ), file, 448
            )
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    private fun stopRecord(cancel: Boolean) {
        wavRecorder?.stopRecording()
        wavRecorder = null
        bottomPanel.visibility = View.GONE
        if (cancel) {
            audioFile = null
        }
        refreshTopScrollContainer()
    }

    private fun startRecord() {
        bottomPanel.visibility = View.VISIBLE

        val file = File(filesDir, "audio")
        if (!file.exists()) {
            file.mkdirs()
        }
        audioFile =
            File(file, "audio_${System.currentTimeMillis()}.wav")
        Log.d(TAG, "audioFile: ${audioFile!!.absolutePath}")
        wavRecorder = WavRecorder()

        wavRecorder?.startRecording(audioFile!!)
    }

    private fun clearHistory() {
        if (isLoadLlmModel) {
            chatList.clear()
            modelScope.launch {
                llmWrapper.reset()
            }
        }
        if (isLoadVlmModel) {
            vlmChatList.clear()
            modelScope.launch {
                vlmWrapper.reset()
            }
        }
        messages.clear()
        audioFile = null
        clearImages()
        reloadRecycleView()
    }

    private var popupWindow: PopupWindow? = null
    private fun showPopupMenu(anchorView: View) {
        if (popupWindow?.isShowing == true) {
            popupWindow?.dismiss()
            return
        }

        val popupView = LayoutInflater.from(this).inflate(R.layout.menu_layout, null)

        popupWindow = PopupWindow(
            popupView,
            anchorView.width * 2,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
            true
        )

        popupWindow?.isOutsideTouchable = true
        popupWindow?.elevation = 10f

        val btnCamera = popupView.findViewById<Button>(R.id.btn_camera)
        val btnPhoto = popupView.findViewById<Button>(R.id.btn_photo)

        btnCamera.setOnClickListener {
            popupWindow?.dismiss()
            checkAndOpenCamera()
        }
        btnPhoto.setOnClickListener {
            popupWindow?.dismiss()
            openGallery()
        }

        popupView.measure(
            View.MeasureSpec.UNSPECIFIED,
            View.MeasureSpec.UNSPECIFIED
        )
        val popupHeight = popupView.measuredHeight
        popupWindow?.showAsDropDown(anchorView, 0, -anchorView.height - popupHeight)
    }

    private var photoUri: Uri? = null
    private var photoFile: File? = null

    private fun checkAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                2001
            )
        } else {
            openCamera()
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = File(
            getExternalFilesDir(Environment.DIRECTORY_PICTURES),
            "photo_${System.currentTimeMillis()}.jpg"
        )
        photoUri = FileProvider.getUriForFile(
            this,
            "${applicationContext.packageName}.fileprovider",
            photoFile!!
        )

        intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
        intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        startActivityForResult(intent, 1001)
    }

    private fun clearImages() {
        savedImageFiles.clear()
        refreshTopScrollContainer()
    }

    private fun refreshTopScrollContainer() {
        runOnUiThread {
            topScrollContainer.removeAllViews()
            if (savedImageFiles.isEmpty() && audioFile == null) {
                scrollImages.visibility = View.GONE
                return@runOnUiThread
            }

            scrollImages.visibility = View.VISIBLE

            for (file in savedImageFiles) {
                val itemView = LayoutInflater.from(this)
                    .inflate(R.layout.item_image_scroll, topScrollContainer, false)
                val ivImage = itemView.findViewById<ImageView>(R.id.iv_image)
                val btnRemove = itemView.findViewById<ImageButton>(R.id.btn_remove)

                ivImage.setImageURI(Uri.fromFile(file))

                btnRemove.setOnClickListener {
                    savedImageFiles.remove(file)
                    refreshTopScrollContainer()
                }
                topScrollContainer.addView(itemView)
            }

            if (audioFile != null) {
                val audioView = LayoutInflater.from(this)
                    .inflate(R.layout.item_audio_scroll, topScrollContainer, false)
                val audioName = audioView.findViewById<TextView>(R.id.tv_audio_name)
                val audioType = audioView.findViewById<TextView>(R.id.tv_audio_type)
                val btnRemove = audioView.findViewById<ImageButton>(R.id.btn_audio_remove)
                audioName.text = audioFile!!.name
                // TODO: hard code
                audioType.text = "wav"

                btnRemove.setOnClickListener {
                    audioFile = null
                    refreshTopScrollContainer()
                }
                topScrollContainer.addView(audioView)
            }
        }
    }

    private fun reloadRecycleView() {
        adapter.notifyDataSetChanged()
        binding.rvChat.scrollToPosition(messages.size - 1)
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
