#include "editingspinbox.h"


EditingSpinBox::EditingSpinBox(QWidget* parent) : QSpinBox(parent) {
	connect(this, &EditingSpinBox::editingFinished, this, &EditingSpinBox::editingFinishedRelay);
}

EditingSpinBox::~EditingSpinBox() {

}

void EditingSpinBox::stepBy(int steps) {
	int oriVal = value();

	QSpinBox::stepBy(steps);

	if (oriVal!=value()) {
		emit contentUpdated();
	}
}

void EditingSpinBox::editingFinishedRelay() {
	emit contentUpdated();
}
