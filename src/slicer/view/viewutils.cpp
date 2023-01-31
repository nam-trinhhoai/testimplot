#include "viewutils.h"

std::vector<ViewType> allViewTypes() {
	std::vector<ViewType> viewTypes{ViewType::InlineView, ViewType::XLineView, ViewType::BasemapView,
				ViewType::StackBasemapView, ViewType::View3D, ViewType::RandomView};
	return viewTypes;
}
