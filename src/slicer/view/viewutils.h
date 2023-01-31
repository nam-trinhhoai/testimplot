#ifndef VIEWUTILS_H_
#define VIEWUTILS_H_

#include <vector>

typedef enum e_ViewType {
	InlineView,
	XLineView,
	BasemapView,
	StackBasemapView,
	View3D,
	RandomView,
	UndefinedView
}
ViewType;

typedef enum e_SampleUnit {
	NONE,
	TIME,
	DEPTH
} SampleUnit;

std::vector<ViewType> allViewTypes();

#endif
