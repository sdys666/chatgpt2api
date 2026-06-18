from services.openai_backend_api import OpenAIBackendAPI


def test_text_attachment_json_detection_ignores_python_list_snippet():
    answer = '''for pat in ["故事板图片提示词", "线稿", "灰", "单张", "Prompt", "storyboard"]:
    print(pat)
'''
    assert not OpenAIBackendAPI._is_text_attachment_complete_json_answer(answer)


def test_text_attachment_json_detection_accepts_json_object():
    answer = '{"shots":[{"title":"故事板 02A","data":{"shotNo":"02A"}}]}'
    assert OpenAIBackendAPI._is_text_attachment_complete_json_answer(answer)
