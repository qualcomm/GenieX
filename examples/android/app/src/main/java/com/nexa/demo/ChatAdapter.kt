// Copyright 2024-2026 Nexa AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.nexa.demo

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.text.method.LinkMovementMethod
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon
import io.noties.markwon.ext.latex.JLatexMathPlugin
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.inlineparser.MarkwonInlineParserPlugin
import io.noties.markwon.linkify.LinkifyPlugin
import java.io.File


data class Message(
    val content: String,
    val type: MessageType,
    val images: List<File> = emptyList(),
    val audio: List<File> = emptyList()
)

enum class MessageType(val value: Int) {
    USER(0),
    ASSISTANT(1),
    PROFILE(2),
    IMAGES(3),
    ASSISTANT_IMAGES(4);

    companion object {
        fun from(value: Int): MessageType =
            entries.firstOrNull { it.value == value } ?: PROFILE
    }
}

class ChatAdapter(private val messages: List<Message>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    override fun getItemViewType(position: Int): Int {
        val message = messages[position]
        return message.type.value
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val type = MessageType.from(viewType)
        return when (type) {
            MessageType.USER -> UserViewHolder(inflater.inflate(R.layout.item_user_message, parent, false))
            MessageType.ASSISTANT -> AiViewHolder(inflater.inflate(R.layout.item_ai_message, parent, false))
            MessageType.IMAGES -> ImagesViewHolder(inflater.inflate(R.layout.item_image_message, parent, false))
            MessageType.ASSISTANT_IMAGES -> ImagesViewHolder(
                inflater.inflate(
                    R.layout.item_assistant_image_message,
                    parent,
                    false
                )
            )

            else -> ProfileViewHolder(inflater.inflate(R.layout.item_profile_message, parent, false))
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        if (holder is UserViewHolder) holder.bind(message)
        if (holder is AiViewHolder) holder.bind(message)
        if (holder is ImagesViewHolder) holder.bind(message)
        if (holder is ProfileViewHolder) holder.bind(message)
    }

    override fun getItemCount() = messages.size

    class UserViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvMessage: TextView = itemView.findViewById(R.id.tv_message)
        fun bind(message: Message) {
            tvMessage.text = message.content
        }
    }

    class AiViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvMessage: TextView = itemView.findViewById(R.id.tv_message)
        private val markwon: Markwon = Markwon.builder(itemView.context)
            .usePlugin(StrikethroughPlugin.create())
            .usePlugin(TablePlugin.create(itemView.context))
            .usePlugin(LinkifyPlugin.create())
            .usePlugin(MarkwonInlineParserPlugin.create())
            .usePlugin(JLatexMathPlugin.create(tvMessage.textSize) { builder ->
                builder.inlinesEnabled(true)
                builder.blocksEnabled(true)
            })
            .build()

        fun bind(message: Message) {
            markwon.setMarkdown(tvMessage, message.content.trim())
            tvMessage.movementMethod = LinkMovementMethod.getInstance()
        }
    }

    class ProfileViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvMessage: TextView = itemView.findViewById(R.id.tv_message)

        fun bind(message: Message) {
            tvMessage.text = message.content
        }

        private fun dpToPx(dp: Int, context: android.content.Context): Int {
            return (dp * context.resources.displayMetrics.density).toInt()
        }
    }

    class ImagesViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageContainer: LinearLayout = itemView.findViewById(R.id.image_container)
        fun bind(message: Message) {
            val savedImageFiles = message.images
            imageContainer.removeAllViews()
            val context = itemView.context

            for (file in savedImageFiles) {
                val itemView = LayoutInflater.from(context)
                    .inflate(R.layout.item_image_item_message, imageContainer, false)
                val ivImage = itemView.findViewById<ImageView>(R.id.iv_image)
                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                if (bitmap != null) {
                    ivImage.setImageBitmap(bitmap)
                }
                imageContainer.addView(itemView)
            }
        }
    }
}
