#pragma once

#include "../core/session.h"

namespace ember {

// RAII guard to temporarily set a slot's cur_pos and restore it on scope exit.
class CurPosGuard {
public:
    CurPosGuard() = default;
    CurPosGuard(const CurPosGuard&) = delete;
    CurPosGuard& operator=(const CurPosGuard&) = delete;
    CurPosGuard(CurPosGuard&& other) noexcept { *this = std::move(other); }
    CurPosGuard& operator=(CurPosGuard&& other) noexcept {
        if (this == &other) return *this;
        restore();
        session_ = other.session_;
        slot_ = other.slot_;
        saved_ = other.saved_;
        active_ = other.active_;
        other.session_ = nullptr;
        other.active_ = false;
        return *this;
    }

    ~CurPosGuard() { restore(); }

    static CurPosGuard set(Session& session, int slot, int new_pos) {
        CurPosGuard g;
        g.session_ = &session;
        g.slot_ = slot;
        g.saved_ = session.cur_pos(slot);
        session.set_cur_pos(slot, new_pos);
        g.active_ = true;
        return g;
    }

    void restore() {
        if (!active_ || !session_) return;
        session_->set_cur_pos(slot_, saved_);
        active_ = false;
    }

private:
    Session* session_ = nullptr;
    int slot_ = 0;
    int saved_ = 0;
    bool active_ = false;
};

}  // namespace ember

