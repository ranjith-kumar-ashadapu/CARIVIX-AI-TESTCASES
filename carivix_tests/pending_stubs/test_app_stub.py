"""
App Developers – Test Stubs
==============================

Status : PENDING – App (mobile/desktop/PWA) developer team delivery
TC-IDs : TC-APP-01 .. TC-APP-08  (placeholder – IDs to be confirmed)

These tests cover native/web app installation, offline behaviour, push notifications,
and performance on mobile devices.  All are @pytest.mark.skip until the app team delivers.

Run stubs:
    pytest carivix_tests/pending_stubs/test_app_stub.py -v
"""

import pytest

_REASON = "PENDING – App team delivery: mobile/PWA app not yet available"

APP_URL = "http://localhost:3000"   # PWA or web-app entry point (update when live)


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app01_pwa_manifest_is_valid():
    """
    TC-APP-01 – GET /manifest.json returns a valid Progressive Web App manifest.

    Expected:
        HTTP 200.
        Fields: name, short_name, start_url, display, icons (≥1).

    TODO: Fetch /manifest.json and validate required PWA fields.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app02_service_worker_registered():
    """
    TC-APP-02 – Service Worker is registered and activated in the browser.

    Expected: navigator.serviceWorker.ready resolves without error.

    TODO: Use Playwright page.evaluate to check SW registration state.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app03_offline_mode_shows_cached_dashboard():
    """
    TC-APP-03 – Disabling network in the browser still serves the cached dashboard shell.

    Expected: Page loads from SW cache; no network error screen.

    TODO: page.context.set_offline(True) → navigate to dashboard → assert content visible.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app04_push_notification_permission_requested():
    """
    TC-APP-04 – On first load, the app requests push notification permission.

    Expected: Browser permission dialog appears (or is denied gracefully).

    TODO: Grant/deny permission via Playwright context and assert app handles both.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app05_first_contentful_paint_under_2s():
    """
    TC-APP-05 – First Contentful Paint (FCP) < 2000 ms on a throttled connection.

    Expected: Measured via Performance API.

    TODO:
        Use Playwright's tracing or performance.timing API.
        Assert FCP metric < 2000.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app06_add_to_home_screen_prompt_shown():
    """
    TC-APP-06 – 'Add to Home Screen' prompt is triggered per PWA install criteria.

    Expected: beforeinstallprompt event fires on eligible browsers.

    TODO: Intercept the beforeinstallprompt event via page.evaluate listener.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app07_app_state_persists_after_reload():
    """
    TC-APP-07 – User's query history / session state survives a page reload.

    Expected: After page.reload(), previous query results or session tokens
    are still present in localStorage / IndexedDB.

    TODO: Set state → reload → assert state persists.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_app08_accessibility_aria_labels_present():
    """
    TC-APP-08 – Key interactive elements have ARIA labels for screen-reader support.

    Expected:
        All buttons have aria-label or accessible text.
        Form inputs have associated <label> or aria-labelledby.

    TODO: Use page.locator("button").all() and assert each has an accessible name.
    """
    raise NotImplementedError
