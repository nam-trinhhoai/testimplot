#include "iterationlistwidgetitem.h"

IterationListWidgetItem::IterationListWidgetItem(const IterationListWidgetItem& other) : QListWidgetItem(other)
{}

IterationListWidgetItem::IterationListWidgetItem(const QIcon& icon, const QString& text, QListWidget *parent, int type) :
    QListWidgetItem(icon, text, parent, type)
{
    this->setData(Qt::UserRole+1, 0);
}

IterationListWidgetItem::IterationListWidgetItem(const QString& text, QListWidget *parent, int type) :
    QListWidgetItem(text, parent, type)
{
    this->setData(Qt::UserRole+1, 0);
}

IterationListWidgetItem::IterationListWidgetItem(QListWidget *parent, int type) :
    QListWidgetItem(parent, type)
{
    this->setData(Qt::UserRole+1, 0);
}

bool IterationListWidgetItem::operator<(const QListWidgetItem &other) const {
    bool test;
    int other_val = other.data(Qt::UserRole+1).toInt(&test);
    if (test) {
        int val = this->data(Qt::UserRole+1).toInt(&test);
        return val<other_val;
    } else {
        return QListWidgetItem::operator<(other);
    }
}
