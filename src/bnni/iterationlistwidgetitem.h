#ifndef ITERATIONLISTWIDGETITEM_H
#define ITERATIONLISTWIDGETITEM_H

#include <QListWidgetItem>

class IterationListWidgetItem : public QListWidgetItem
{
public:
    explicit IterationListWidgetItem(const IterationListWidgetItem& other);
    explicit IterationListWidgetItem(const QIcon& icon, const QString& text, QListWidget *parent = 0, int type = Type);
    explicit IterationListWidgetItem(const QString& text, QListWidget *parent = 0, int type = Type);
    explicit IterationListWidgetItem(QListWidget *parent = 0, int type = Type);
    virtual bool operator<(const QListWidgetItem& other) const override;
//    virtual bool operator<(const QListWidgetItem &other) const override;
};

#endif // ITERATIONLISTWIDGETITEM_H
