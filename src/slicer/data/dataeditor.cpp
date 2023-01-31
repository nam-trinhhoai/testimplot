#include "dataeditor.h"
#include "editabledata.h"

DataEditor::DataEditor(EditableData* data, QObject* parent) :
        QObject(parent), m_data(data)  {
}

DataEditor::~DataEditor() {}

bool DataEditor::isEditing() const {
    return m_data->currentModifier()==this;
}

bool DataEditor::startEditing() {
    return m_data->setCurrentModifier(this);
}

