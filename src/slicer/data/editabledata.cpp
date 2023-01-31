#include "editabledata.h"
#include "dataeditor.h"

#include <QMutexLocker>

EditableData::EditableData(WorkingSetManager* manager, QObject* parent) :
        IData(manager, parent) {
    m_currentModifier = nullptr;
}

EditableData::~EditableData() {}

bool EditableData::isEdited() const {
    QMutexLocker lock(&m_modifierMutex);
    return m_currentModifier!=nullptr;
}

DataEditor* EditableData::currentModifier() const {
    QMutexLocker lock(&m_modifierMutex);
    return m_currentModifier;
}

bool EditableData::setCurrentModifier(DataEditor* editor) {
    QMutexLocker lock(&m_modifierMutex);

    bool result;
    if (m_currentModifier==nullptr) {
        QObject::connect(editor, &DataEditor::editionFinished, 
                         this, &EditableData::resetModifier);
        m_currentModifier = editor;
        result = true;
    } else {
        result = false;
    }
}

void EditableData::resetModifier() {
    QMutexLocker lock(&m_modifierMutex);
    QObject::disconnect(m_currentModifier, &DataEditor::editionFinished,
                         this, &EditableData::resetModifier);
    m_currentModifier = nullptr;
}

