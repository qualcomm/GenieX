package com.geniex.sdk.jni

import com.geniex.sdk.bean.ProfilingData
import com.geniex.sdk.bean.RerankConfig
import com.geniex.sdk.bean.RerankerCreateInput
import com.geniex.sdk.bean.RerankerOutputResult

class Reranker {
    // Create reranker using RerankerCreateInput (matches ml_RerankerCreateInput)
    external fun create(rerankerCreateInput: RerankerCreateInput): Long
    external fun destroy(handle: Long): Int
    external fun rerank(
        handle: Long,
        query: String,
        documents: Array<String>,
        config: RerankConfig
    ): RerankerOutputResult?

}