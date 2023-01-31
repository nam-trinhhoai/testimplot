#ifndef DATAEDITOR_H
#define DATAEDITOR_H

#include <QObject>

class EditableData;

class DataEditor : public QObject {
Q_OBJECT
public:
    DataEditor(EditableData* data, QObject* parent=0);
    virtual ~DataEditor();

    virtual bool isEditing() const;
    virtual bool startEditing();

signals:
    void editionFinished();

private:
    EditableData* m_data;
};

Q_DECLARE_METATYPE(DataEditor*)

#endif
