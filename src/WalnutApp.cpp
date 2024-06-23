#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/UI/UI.h"

#include "InputType.h"

class ExampleLayer : public Walnut::Layer
{
public:
	virtual void OnUIRender() override
	{
		ImGui::Begin("Input");
		const char* const inputs[] = { "image","video","webcam" };
		static int currentItemInput = 0;
		ImGui::ListBox("Input", &currentItemInput, inputs, 3, 3);

		static char filePath[150] = ""; ImGui::InputText("Path to the file or camera number", filePath, 150, ImGuiInputTextFlags_CharsNoBlank);


		const char* const functions[] = { "correction","doc scanner","plate scanner", "face detection", "pose detection", "hand detection", "selective search" };
		static int currentItemFunc = 0;
		ImGui::ListBox("Functions", &currentItemFunc, functions, 7, 7);

		if (ImGui::Button("Run")) {
			if (currentItemInput == 0) {
				importAndSaveImage(filePath, currentItemFunc);
			}
			else if (currentItemInput == 1) {
				importAndSaveVideo(filePath, currentItemFunc);
			}
			else {
				importAndSaveWebcam(atoi(filePath), currentItemFunc);
			}
		}

		ImGui::End();
	}

private:
	bool m_AboutModalOpen = false;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "L0_0KER";
	spec.CustomTitlebar = true;

	Walnut::Application* app = new Walnut::Application(spec);
	std::shared_ptr<ExampleLayer> exampleLayer = std::make_shared<ExampleLayer>();
	app->PushLayer(exampleLayer);
	return app;
}