/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers_cpp.h
 * \brief A C++ binding to common set of tokenizers
 */
 #ifndef TOKENIZERS_CPP_H_
 #define TOKENIZERS_CPP_H_
 
 #include <memory>
 #include <string>
 #include <vector>
 
 namespace tokenizers {
 
 /*!
  * \brief a universal tokenizer that loads
  *  either HF's tokenizer or sentence piece,
  *  depending on the constructor
  */
 class Tokenizer {
  public:
   /*! \brief virtual destructor */
   virtual ~Tokenizer() {}
 
   /*!
    * \brief Encode text into ids.
    * \param text The input text.
    * \returns The encoded token ids.
    */
   virtual std::vector<int32_t> Encode(const std::string& text) = 0;
 
   /*!
    * \brief Encode a batch of texts into ids.
    * \param texts The input texts.
    * \returns The encoded token ids.
    */
   virtual std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts) {
     // Fall back when the derived class does not implement this function.
     std::vector<std::vector<int32_t>> ret;
     ret.reserve(texts.size());
     for (const auto& text : texts) {
       ret.push_back(Encode(text));
     }
     return ret;
   }
 
   /*!
    * \brief Decode token ids into text.
    * \param text The token ids.
    * \returns The decoded text.
    */
   virtual std::string Decode(const std::vector<int32_t>& ids) = 0;
 
   /*!
    * \brief Returns the vocabulary size. Special tokens are considered.
    */
   virtual size_t GetVocabSize() = 0;
 
   /*!
    * \brief Convert the given id to its corresponding token if it exists. If not, return an
    * empty string.
    */
   virtual std::string IdToToken(int32_t token_id) = 0;
 
   /*!
    * \brief Convert the given token to its corresponding id if it exists. If not, return -1.
    */
   virtual int32_t TokenToId(const std::string& token) = 0;
 
   //---------------------------------------------------
   // Factory functions from byte-blobs
   // These factory function takes in in-memory blobs
   // so the library can be independent from filesystem
   //---------------------------------------------------
   /*!
    * \brief Create HF tokenizer from a single json file.
    *
    * \param json_path The path to the json file.
    * \return The created tokenzier.
    */
   static std::unique_ptr<Tokenizer> FromJSON(const std::string& json_path);
   /*!
    * \brief Create HF tokenizer from a single in-memory json blob.
    *
    * \param json_blob The json blob.
    * \return The created tokenzier.
    */
   static std::unique_ptr<Tokenizer> FromBlobJSON(const std::string& json_blob);
   /*!
    * \brief Create BPE tokenizer
    *
    * \param vocab_blob The blob that contains vocabs.
    * \param merges_blob The blob that contains the merges.
    * \param added_tokens The added tokens.
    * \return The created tokenizer.
    */
   static std::unique_ptr<Tokenizer> FromBlobByteLevelBPE(const std::string& vocab_blob,
                                                          const std::string& merges_blob,
                                                          const std::string& added_tokens = "");
   /*!
    * \brief Create SentencePiece.
    *
    * \param model_blob The blob that contains vocabs.
    * \return The created tokenizer.
    */
   static std::unique_ptr<Tokenizer> FromBlobSentencePiece(const std::string& model_blob);
   /*!
    * \brief Create RWKVWorldTokenizer.
    *
    * \param model_blob The blob that contains vocabs.
    * \return The created tokenizer.
    */
   static std::unique_ptr<Tokenizer> FromBlobRWKVWorld(const std::string& model_blob);
 };
 
 }  // namespace tokenizers
 #endif  // TOKENIZERS_CPP_H_
 