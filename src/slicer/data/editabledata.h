#ifndef EDITABLEDATA_H
#define EDITABLEDATA_H

#include <QObject>
#include <QMutex>
#include "idata.h"

class DataEditor;

class EditableData : public IData {
Q_OBJECT
public:
    EditableData(WorkingSetManager* manager, QObject* parent=0);
    virtual ~EditableData();

    virtual bool isEdited() const;
    virtual DataEditor* currentModifier() const;
    virtual bool setCurrentModifier(DataEditor* editor);

private:
    void resetModifier();

    DataEditor* m_currentModifier;
    mutable QMutex m_modifierMutex;
};

Q_DECLARE_METATYPE(EditableData*)

#endif
