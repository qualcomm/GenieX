# Test asset attributions

## quality_dog.jpg

- **Subject:** Golden retriever portrait used by VLM keyword-quality tests
  ([test_llama_cpp_vlm.py](../plugins/llama_cpp/test_llama_cpp_vlm.py),
  [test_qairt_vlm.py](../plugins/qairt/test_qairt_vlm.py)).
- **Source:** "Golden Retriever Carlos" by Dirk Vorderstraße,
  https://commons.wikimedia.org/wiki/File:Golden_Retriever_Carlos_(10581910556).jpg
- **License:** Creative Commons Attribution 2.0 Generic (CC BY 2.0),
  https://creativecommons.org/licenses/by/2.0/
- **Modifications:** Resampled from 1500×1000 to 512×341 and re-encoded as
  JPEG (q≈75) to keep the repo asset under 50 KB.

This file mirrors the same image the upstream `test-llama.cpp` QDC scorecard
fetches at runtime (`scripts/snapdragon/qdc/tests/run_scorecard_posix.py`,
`_VLM_TEST_IMAGE_URL`); we ship a downscaled copy so the offline pytest run
needs no network for VLM quality checks.
